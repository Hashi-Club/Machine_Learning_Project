from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import math

# Import Strategy Base and Concrete Strategies
try:
    from .strategy_base import Strategy, StrategyResult
    from .basic_strategies import MovingAverageCross, RSIReversion, BollingerBreakout, BuyAndHold, MACDStrategy
    from .ml_strategies import XGBoostStrategy
except ImportError:
    from strategy_base import Strategy, StrategyResult
    from basic_strategies import MovingAverageCross, RSIReversion, BollingerBreakout, BuyAndHold, MACDStrategy
    from ml_strategies import XGBoostStrategy

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


STRATEGY_MAP = {
    'buy_hold': BuyAndHold,
    'ma_cross': MovingAverageCross,
    'rsi_reversion': RSIReversion,
    'boll_breakout': BollingerBreakout,
    'macd': MACDStrategy,
    'xgboost': XGBoostStrategy,
}


# ============================= Backtest Engine =============================
def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    apply_fees: bool = False,
    capital: float = 100000.0,
    max_position: float = 1.0,
) -> StrategyResult:
    """运行单策略回测，支持可选港股手续费扣除。

    持仓支持部分仓位：策略返回 -max_position~max_position 之间的浮点权重。
    手续费按交易金额计算，并转换为对总资本的收益影响。
    """
    # 1. 准备数据
    raw_positions = strategy.generate_positions(df).astype(float)
    positions = raw_positions.reindex(df.index).fillna(0.0)
    max_abs = abs(max_position)
    if max_abs <= 0:
        raise ValueError('max_position 必须为正数')
    positions = positions.clip(lower=-max_abs, upper=max_abs)
    close = df['close']

    # 2. 初始化回测状态
    actual_positions = pd.Series(0.0, index=df.index) # 实际持仓权重 (基于手数)
    fee_series = pd.Series(0.0, index=df.index)
    trades = []
    
    current_shares = 0 # 当前持股数 (股)
    current_cash = capital # 当前现金（名义资金，随交易与手续费变动）
    avg_entry_price: Optional[float] = None
    month_order_counts: Dict[pd.Timestamp, int] = {}
    
    prev_signal_weight = 0.0 # 上一时刻的信号权重

    # 3. 逐日迭代 (模拟交易)
    for dt in df.index:
        price = close.loc[dt]
        signal_weight = positions.loc[dt]
        
        # 检查信号是否发生变化 (使用小容差处理浮点数)
        # 注意：这里假设策略信号变化才触发调仓，而不是每天根据价格波动再平衡
        if abs(signal_weight - prev_signal_weight) > 1e-6:
            # 计算目标持仓（不允许做空）：正信号 -> 目标为非负仓位；非正信号 -> 清仓
            if signal_weight > 0:
                # 目标是尽可能接近权重 signal_weight * capital 的仓位，但受现金与手续费约束
                target_val = signal_weight * capital
                # 初步目标股数（整手）
                prelim_shares = math.floor((target_val / price) / 100) * 100
                # 受现金约束的可买股数（整手）：考虑手续费后不超过 current_cash
                # 使用二分查找最大整手可买
                low, high = 0, max(prelim_shares, 0)
                max_affordable = 0
                while low <= high:
                    mid = (low + high) // 2
                    amount = mid * price
                    fees_total = compute_hk_fees(amount=amount, date=dt)['total_fee'] if apply_fees else 0.0
                    total_cost = amount + fees_total
                    if total_cost <= current_cash:
                        max_affordable = mid
                        low = mid + 100
                    else:
                        high = mid - 100
                target_shares = int(max_affordable)
            else:
                target_shares = 0

            # 计算需要交易的股数（不允许卖出超过当前持股，不做空）
            desired_diff = target_shares - current_shares
            if desired_diff > 0:
                # 买入：按整手，且不超过现金能买的最大股数
                buy_amount = desired_diff * price
                fees_total = compute_hk_fees(amount=buy_amount, date=dt)['total_fee'] if apply_fees else 0.0
                total_cost = buy_amount + fees_total
                if total_cost > current_cash:
                    # 进一步缩减购买股数直到可支付
                    # 由于手续费随金额近似线性增长，按整手递减
                    reduce = desired_diff
                    while reduce > 0:
                        test_shares = reduce
                        test_amount = test_shares * price
                        test_fees = compute_hk_fees(amount=test_amount, date=dt)['total_fee'] if apply_fees else 0.0
                        if test_amount + test_fees <= current_cash:
                            desired_diff = test_shares
                            break
                        reduce -= 100
                    else:
                        desired_diff = 0
            else:
                # 卖出：最多卖到0股
                desired_diff = max(desired_diff, -current_shares)

            diff_shares = desired_diff
            if diff_shares != 0:
                trade_val = abs(diff_shares) * price
                trade_type = 'BUY' if diff_shares > 0 else 'SELL'
                month_start = pd.Timestamp(dt.year, dt.month, 1)
                prev_count = month_order_counts.get(month_start, 0)
                trade_record = {
                    'date': dt,
                    'type': trade_type,
                    'price': price,
                    'shares': float(diff_shares),
                    'amount': float(trade_val),
                    'target_shares': float(target_shares),
                }

                # 手续费并更新现金
                fees_total = compute_hk_fees(amount=trade_val, date=dt)['total_fee'] if apply_fees else 0.0
                if diff_shares > 0:
                    # 买入支付现金
                    total_cost = trade_val + fees_total
                    current_cash -= total_cost
                else:
                    # 卖出收回现金（卖出也可能产生费用，这里费用从收益中扣，现金收到净额）
                    current_cash += (trade_val - fees_total)

                # 费用影响加入收益序列（基于名义 capital）
                if apply_fees and fees_total > 0:
                    fee_series.loc[dt] -= fees_total / capital

                # 更新均价与记录
                if diff_shares > 0: # 买入
                    if current_shares <= 0 or avg_entry_price is None:
                        avg_entry_price = price
                    else:
                        total_shares = current_shares + diff_shares
                        if total_shares > 0:
                            avg_entry_price = (avg_entry_price * current_shares + price * diff_shares) / total_shares
                    trade_record['avg_entry_price'] = avg_entry_price
                else: # 卖出
                    if avg_entry_price is not None and current_shares != 0:
                        realized_pct = price / avg_entry_price - 1
                        trade_record['pnl_pct'] = realized_pct
                    if target_shares <= 0:
                        avg_entry_price = None

                trades.append(trade_record)
                month_order_counts[month_start] = prev_count + 1
                current_shares += diff_shares

            prev_signal_weight = signal_weight
        
        # 更新当日实际持仓权重 (用于计算次日收益)
        # 权重 = (持股数 * 当前价格) / 初始资金
        # 注意：这里使用初始资金作为分母，意味着不复利再投资 (Simple Return on Capital)
        # 如果希望模拟复利，分母应为当前权益，但这会引入循环依赖。
        # 鉴于 capital 是名义资金，这种做法是标准的回测简化。
        actual_positions.loc[dt] = (current_shares * price) / capital

    # 4. 计算收益指标
    daily_ret = close.pct_change().fillna(0)
    # 策略收益 = 每日收益率 * 昨日持仓权重
    strat_ret = daily_ret * actual_positions.shift(1).fillna(0)
    
    # 结合手续费后的策略收益
    net_returns = strat_ret + fee_series  # fee_series 为负值
    equity = (1 + net_returns).cumprod()
    trades_df = pd.DataFrame(trades)

    metrics = compute_metrics(net_returns, equity, trades_df)
    return StrategyResult(name=strategy.name, metrics=metrics, equity=equity, positions=actual_positions, trades=trades_df)


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
def compute_hk_fees(amount: float,
                    date: Optional[pd.Timestamp] = None) -> Dict[str, float]:
    """计算港股单笔订单所有费用 (仅股票, 固定费率).

    参数:
        amount: 成交金额 (成交价 * 股数)
        date: 交易日期, 用于判断某些费用是否生效 (系统使用费取消时间等)

    返回: dict 各项费用与总额 total_fee
    """
    if date is None:
        date = pd.Timestamp.today()

    # 1. 佣金: 0.03% * amount, 最低 3 HKD
    commission = max(3.0, amount * 0.0003)

    # 2. 平台使用费: 固定 15 HKD
    platform_fee = 15.0

    # 3. 系统使用费: 2023-01-01 起取消
    sys_fee = 0.5 if date < pd.Timestamp('2023-01-01') else 0.0

    # 4. 印花税 (股票): 0.1% * amount, 向上取整, 最低 1 HKD
    stamp_duty = max(1.0, math.ceil(amount * 0.001))

    # 5. 其他杂费 (结算 + 交易 + 证监会 + 财汇局)
    # 结算交收费 0.0042%
    settlement_fee = amount * 0.000042
    # 交易费 0.00565% (min 0.01)
    trading_fee = max(0.01, amount * 0.0000565)
    # 证监会征费 0.0027% (min 0.01)
    sfc_levy = max(0.01, amount * 0.000027)
    # 财汇局征费 0.00015%
    frc_levy = amount * 0.0000015

    # 汇总
    fees = {
        'commission': round(commission, 2),
        'platform_fee': round(platform_fee, 2),
        'system_fee': round(sys_fee, 2),
        'stamp_duty': round(stamp_duty, 2),
        'settlement_fee': round(settlement_fee, 2),
        'trading_fee': round(trading_fee, 2),
        'sfc_levy': round(sfc_levy, 2),
        'frc_levy': round(frc_levy, 4),
    }
    fees['total_fee'] = round(sum(fees.values()), 2)
    return fees

# ============================= CLI Runner =============================
def parse_args():
    p = argparse.ArgumentParser(description='多策略回测框架')
    p.add_argument('--file', default='data/data_with_indicators.txt', help='数据文件路径 (含OHLC)')
    p.add_argument('--start', default='2024-01-01', help='起始日期 YYYY-MM-DD')
    p.add_argument('--end', default='2025-04-24', help='结束日期 YYYY-MM-DD')
    p.add_argument('--strategies', nargs='+', default=list(STRATEGY_MAP.keys()), help='策略名称列表')
    p.add_argument('--output', help='结果保存CSV文件名')
    p.add_argument('--apply-fees', action='store_true', help='启用港股手续费扣除')
    p.add_argument('--capital', type=float, default=100000.0, help='名义资金（用于手续费金额计算）')
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
        # 调用 prepare 方法 (如果需要)
        strat.prepare(str(path))
        
        res = run_backtest(
            df,
            strat,
            apply_fees=args.apply_fees,
            capital=args.capital,
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
