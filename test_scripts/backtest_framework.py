from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import math


# ============================= Data Loading =============================
def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', engine='python')
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' not in df.columns:
        raise ValueError("数据文件缺少 date 列")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少 {col} 列")
    return df


# ============================= Strategy Base =============================
@dataclass
class StrategyResult:
    name: str
    metrics: Dict[str, float]
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame


class Strategy:
    name: str = "Base"

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """返回每日持仓权重 ([-1, 1] 浮点)，index 与 df 一致。"""
        raise NotImplementedError


# ============================= Concrete Strategies =============================
class MovingAverageCross(Strategy):
    name = "ma_cross"

    def __init__(self, fast: int = 5, slow: int = 20):
        self.fast = fast
        self.slow = slow

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        fast_ma = close.rolling(self.fast, min_periods=1).mean()
        slow_ma = close.rolling(self.slow, min_periods=1).mean()
        signal = (fast_ma > slow_ma).astype(int)
        return signal


class RSIReversion(Strategy):
    name = "rsi_reversion"

    def __init__(self, period: int = 14, low: float = 30.0, high: float = 70.0):
        self.period = period
        self.low = low
        self.high = high

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        # 如果已有 rsi14 列，复用；否则计算
        if 'rsi14' in df.columns and df['rsi14'].notna().any() and self.period == 14:
            rsi = df['rsi14']
        else:
            close = df['close']
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(self.period).mean()
            loss = (-delta.clip(upper=0)).rolling(self.period).mean()
            rs = gain / (loss.replace(0, np.nan))
            rsi = 100 - (100 / (1 + rs))
        pos = pd.Series(0, index=df.index)
        pos[rsi < self.low] = 1
        pos[rsi > self.high] = 0
        return pos.ffill().fillna(0)


class BollingerBreakout(Strategy):
    name = "boll_breakout"

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        if all(c in df.columns for c in ['boll_mid', 'boll_upper', 'boll_lower']):
            mid = df['boll_mid']
            upper = df['boll_upper']
            lower = df['boll_lower']
        else:
            mid = close.rolling(self.period, min_periods=1).mean()
            std = close.rolling(self.period, min_periods=1).std()
            upper = mid + self.num_std * std
            lower = mid - self.num_std * std
        signal = pd.Series(0, index=df.index)
        signal[close > upper] = 1  # 突破做多
        signal[close < lower] = 0  # 跌破下轨清仓
        return signal.ffill().fillna(0)


class BuyAndHold(Strategy):
    name = "buy_hold"

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        # 始终持有，权重为 1
        return pd.Series(1.0, index=df.index)


STRATEGY_MAP = {
    'ma_cross': MovingAverageCross,
    'rsi_reversion': RSIReversion,
    'boll_breakout': BollingerBreakout,
    'buy_hold': BuyAndHold,
}


# ============================= Backtest Engine =============================
def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    apply_fees: bool = False,
    capital: float = 100000.0,
    fee_package: str = 'fixed',
    instrument_type: str = 'stock',
    commission_free: bool = False,
    max_position: float = 1.0,
) -> StrategyResult:
    """运行单策略回测，支持可选港股手续费扣除。

    持仓支持部分仓位：策略返回 -max_position~max_position 之间的浮点权重。
    手续费按交易金额计算，并转换为对总资本的收益影响。
    """
    raw_positions = strategy.generate_positions(df).astype(float)
    positions = raw_positions.reindex(df.index).fillna(0.0)
    max_abs = abs(max_position)
    if max_abs <= 0:
        raise ValueError('max_position 必须为正数')
    positions = positions.clip(lower=-max_abs, upper=max_abs)
    close = df['close']
    daily_ret = close.pct_change().fillna(0)
    strat_ret = daily_ret * positions.shift(1).fillna(0)

    fee_series = pd.Series(0.0, index=df.index)
    trades = []
    current_pos = 0.0
    avg_entry_price: Optional[float] = None
    # 统计当月订单数（用于阶梯式平台费）: month -> count
    month_order_counts: Dict[pd.Timestamp, int] = {}

    # diff 首行 NaN 代表首次持仓，直接用原值替代
    pos_change = positions.diff().fillna(positions.iloc[0])
    trade_dates = pos_change[pos_change != 0].index
    for dt in trade_dates:
        change = pos_change.loc[dt]
        if change == 0:
            continue
        month_start = pd.Timestamp(dt.year, dt.month, 1)
        prev_count = month_order_counts.get(month_start, 0)
        new_pos = current_pos + change
        trade_type = 'BUY' if change > 0 else 'SELL'
        trade_amount = abs(change) * capital

        price_today = close.loc[dt]
        trade_record = {
            'date': dt,
            'type': trade_type,
            'price': price_today,
            'weight_change': float(change),
            'target_weight': float(new_pos),
        }

        if change > 0:
            if current_pos <= 0 or avg_entry_price is None:
                avg_entry_price = price_today
            else:
                total_weight = current_pos + change
                if total_weight > 0:
                    avg_entry_price = (avg_entry_price * current_pos + price_today * change) / total_weight
            trade_record['avg_entry_price'] = avg_entry_price
        else:
            if avg_entry_price is not None and current_pos != 0:
                realized_pct = price_today / avg_entry_price - 1
                trade_record['pnl_pct'] = realized_pct
            if new_pos <= 0:
                avg_entry_price = None

        trades.append(trade_record)

        if apply_fees and trade_amount > 0:
            fees = compute_hk_fees(
                amount=trade_amount,
                order_count_this_month=prev_count,
                instrument_type=instrument_type,
                date=dt,
                package=fee_package,
                commission_free=commission_free,
            )
            fee_series.loc[dt] -= fees['total_fee'] / capital

        month_order_counts[month_start] = prev_count + 1
        current_pos = new_pos

    # 结合手续费后的策略收益
    net_returns = strat_ret + fee_series  # fee_series 为负值
    equity = (1 + net_returns).cumprod()
    trades_df = pd.DataFrame(trades)

    metrics = compute_metrics(net_returns, equity, trades_df)
    return StrategyResult(name=strategy.name, metrics=metrics, equity=equity, positions=positions, trades=trades_df)


# ============================= Metrics =============================
def compute_metrics(returns: pd.Series, equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
    total_return = equity.iloc[-1] - 1
    num_days = (equity.index[-1] - equity.index[0]).days
    years = max(num_days / 365.25, 1e-9)
    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()
    daily_mean = returns.mean()
    daily_std = returns.std(ddof=0)
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else np.nan
    win_rate = (returns[returns > 0].count() / returns[returns != 0].count()) if returns[returns != 0].count() > 0 else np.nan
    trade_count = len(trades)
    avg_trade_return = trades['pnl_pct'].mean() if 'pnl_pct' in trades.columns and not trades.empty else np.nan
    return {
        'total_return': round(float(total_return), 6),
        'cagr': round(float(cagr), 6),
        'max_drawdown': round(float(max_drawdown), 6),
        'sharpe': round(float(sharpe), 6),
        'win_rate': round(float(win_rate), 6),
        'trade_count': int(trade_count),
        'avg_trade_return': round(float(avg_trade_return), 6),
    }


# ============================= HK Fees Calculation =============================
def _tiered_platform_fee(order_count_this_month: int) -> float:
    """阶梯式平台使用费单笔费用."""
    tiers = [
        (5, 30), (20, 15), (50, 10), (100, 9), (500, 8), (1000, 7),
        (2000, 6), (3000, 5), (4000, 4), (5000, 3), (6000, 2)
    ]
    for limit, fee in tiers:
        if order_count_this_month <= limit:
            return float(fee)
    return 1.0  # 6001 及以上


def compute_hk_fees(amount: float,
                    order_count_this_month: int,
                    instrument_type: str = 'stock',
                    date: Optional[pd.Timestamp] = None,
                    package: str = 'fixed',
                    commission_free: bool = False) -> Dict[str, float]:
    """计算港股单笔订单所有费用.

    参数:
        amount: 成交金额 (成交价 * 股数)
        order_count_this_month: 当前自然月累计订单数 (含本单之前?) 传入本单之前的数量用于阶梯计算
        instrument_type: 'stock' | 'etf' | 'warrant' | 'cbbc' | 'other'
        date: 交易日期, 用于判断某些费用是否生效 (系统使用费取消时间等)
        package: 'fixed' 固定式15港元 / 'tiered' 阶梯式
        commission_free: 免佣期内 True 则佣金为0

    返回: dict 各项费用与总额 total_fee
    """
    if date is None:
        date = pd.Timestamp.today()

    # 佣金: 0.03% * amount, 最低 3 HKD
    if commission_free:
        commission = 0.0
    else:
        commission_raw = amount * 0.0003
        commission = max(3.0, commission_raw)

    # 平台使用费
    if package == 'fixed':
        platform_fee = 15.0
    elif package == 'tiered':
        # 本单属于下一序号, 所以用 (order_count_this_month + 1) 判断档位
        platform_fee = _tiered_platform_fee(order_count_this_month + 1)
    else:
        raise ValueError('package 必须为 fixed 或 tiered')

    # 系统使用费: 每次成交0.50港元 (2023-01-01 起取消)
    sys_fee = 0.5 if date < pd.Timestamp('2023-01-01') else 0.0

    # 结算交收费 0.0042%
    settlement_fee = amount * 0.000042

    # 印花税: 股票 0.1% * amount, 向上取整到整数港元, 最低1; ETF/窝轮/牛熊证免除
    if instrument_type.lower() in {'etf', 'warrant', 'cbbc'}:
        stamp_duty = 0.0
    else:
        stamp_raw = amount * 0.001  # 0.1%
        stamp_duty = max(1.0, math.ceil(stamp_raw))

    # 交易费 0.00565% 最低0.01
    trading_fee_raw = amount * 0.0000565
    trading_fee = max(0.01, trading_fee_raw)

    # 证监会交易征费 0.0027% 最低0.01
    sfc_levy_raw = amount * 0.000027
    sfc_levy = max(0.01, sfc_levy_raw)

    # 财务汇报局征费 0.00015% 无最低说明, 常规四舍五入
    frc_levy = amount * 0.0000015

    # 汇总
    fees = {
        'commission': round(commission, 2),
        'platform_fee': round(platform_fee, 2),
        'system_fee': round(sys_fee, 2),
        'settlement_fee': round(settlement_fee, 2),
        'stamp_duty': round(stamp_duty, 2),
        'trading_fee': round(trading_fee, 2),
        'sfc_levy': round(sfc_levy, 2),
        'frc_levy': round(frc_levy, 4),
    }
    fees['total_fee'] = round(sum(fees.values()), 2)
    return fees


def compute_negative_cash_interest(negative_cash: float, days: int, annual_rate: float = 0.068) -> float:
    """负现金日利息: 单利 = 负现金 * 年利率/365 * 天数."""
    if negative_cash >= 0:
        return 0.0
    interest = (-negative_cash) * annual_rate / 365.0 * days
    return round(interest, 2)


# ============================= CLI Runner =============================
def parse_args():
    p = argparse.ArgumentParser(description='多策略回测框架')
    p.add_argument('--file', default='data/data_with_indicators.txt', help='数据文件路径 (含OHLC)')
    p.add_argument('--start', help='起始日期 YYYY-MM-DD')
    p.add_argument('--end', help='结束日期 YYYY-MM-DD')
    p.add_argument('--strategies', nargs='+', default=['ma_cross', 'rsi_reversion', 'boll_breakout'], help='策略名称列表')
    p.add_argument('--output', help='结果保存CSV文件名')
    p.add_argument('--apply-fees', action='store_true', help='启用港股手续费扣除')
    p.add_argument('--capital', type=float, default=100000.0, help='名义资金（用于手续费金额计算）')
    p.add_argument('--fee-package', choices=['fixed', 'tiered'], default='fixed', help='平台使用费套餐类型')
    p.add_argument('--instrument-type', choices=['stock', 'etf', 'warrant', 'cbbc', 'other'], default='stock', help='产品类型决定印花税等')
    p.add_argument('--commission-free', action='store_true', help='免佣期：佣金设为0')
    p.add_argument('--max-position', type=float, default=1.0, help='持仓权重上限（绝对值）')
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f'数据文件不存在: {path}')

    df = load_price_data(path)
    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]
    if df.empty:
        raise ValueError('筛选后无数据')

    results: List[StrategyResult] = []
    for name in args.strategies:
        if name not in STRATEGY_MAP:
            print(f"跳过未知策略: {name}")
            continue
        strat = STRATEGY_MAP[name]()
        res = run_backtest(
            df,
            strat,
            apply_fees=args.apply_fees,
            capital=args.capital,
            fee_package=args.fee_package,
            instrument_type=args.instrument_type,
            commission_free=args.commission_free,
            max_position=args.max_position,
        )
        results.append(res)

    # 汇总输出
    rows = []
    for r in results:
        row: Dict[str, float] = {'strategy': r.name}  # type: ignore[assignment]
        # 强制将 metrics 转换为 float 以避免类型检查问题
        for k, v in r.metrics.items():
            row[k] = float(v)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    print('\n回测结果汇总:')
    print(summary_df.to_string(index=False))

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f'结果已保存: {args.output}')


if __name__ == '__main__':
    main()
