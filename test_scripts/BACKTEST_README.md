# 回测框架详细文档

文件：`backtest_framework.py`

本框架提供：数据加载、策略接口/示例策略、回测引擎、绩效指标计算以及港股手续费模型集成。

```bash
python .\test_scripts\backtest_framework.py --file data/data_with_indicators.txt --start 2024-01-01 --end 2025-04-24 --strategies ma_cross rsi_reversion boll_breakout buy_hold --apply-fees --capital 100000 --fee-package fixed --max-position 1.0 --output different_methods_with_fees.csv
```

## 目录索引

- 数据加载: `load_price_data()`
- 策略结果数据类: `StrategyResult`
- 策略基类: `Strategy`
- 示例策略: `MovingAverageCross`, `RSIReversion`, `BollingerBreakout`
- 回测主函数: `run_backtest()`
- 指标计算: `compute_metrics()`
- 港股费用计算: `compute_hk_fees()`, `_tiered_platform_fee()`, `compute_negative_cash_interest()`
- CLI 入口与参数: `parse_args()` / `main()`

---

## 数据加载

### `load_price_data(path: Path) -> pd.DataFrame`

读取制表符分隔的行情文件，规范化列名为小写并将 `date` 设为索引。要求最少包含：`open, high, low, close`。

返回的 DataFrame index 为日期（`DatetimeIndex`），按时间升序排序。

错误处理：缺失必要列或日期解析失败会抛出 `ValueError`。

---

## 策略结果数据类

### `@dataclass StrategyResult`

字段说明：

- `name`: 策略名称
- `metrics`: 指标字典（由 `compute_metrics` 生成）
- `equity`: 权益曲线（初始为1，复利叠加）
- `positions`: 每日持仓权重（-1~1 浮点，支持部分仓位；示例策略仅返回0或1）
- `trades`: 交易明细表，含 `date`, `type`（BUY/SELL）, `price`, 可选 `pnl_pct`

---

## 策略基类

### `class Strategy`

接口：`generate_positions(self, df: pd.DataFrame) -> pd.Series`

返回与 `df.index` 对齐的持仓序列。数值语义：

- 1：满仓多头
- 0：空仓
- -1：满仓做空（示例未启用）
- 0~1/0~-1：部分仓位权重

实现新策略只需继承并覆盖该方法。

---

## 示例策略

### `MovingAverageCross(fast=5, slow=20)`

逻辑：`close` 的快速均线高于慢速均线时持有（1），否则空仓（0）。
实现：滚动均值 `rolling().mean()` 对比生成布尔转 int。

### `RSIReversion(period=14, low=30, high=70)`

逻辑：RSI < low 建立多仓；RSI > high 清仓。中间保持前值（`ffill`）。
RSI来源：优先使用已有 `rsi14` 列，否则内部计算（经典平均法）。

### `BollingerBreakout(period=20, num_std=2.0)`

逻辑：收盘价突破上轨 -> 做多；跌破下轨 -> 平仓。中间保持前值。
上/下轨来源：优先使用已有 `boll_mid/boll_upper/boll_lower`，否则内部用均值及标准差计算。

---

## 回测引擎

### `run_backtest(df, strategy, apply_fees=False, capital=100000.0, fee_package='fixed', instrument_type='stock', commission_free=False, max_position=1.0)`

核心步骤：

1. 生成 `positions`（裁剪至 ±`max_position`），移位一日乘以收盘涨跌得日策略收益（信号在收盘形成，下日开盘持仓，简化模型）。
2. 通过 `positions.diff()` 识别权重变化，记录加仓/减仓事件与平均持仓成本。
3. 交易金额 = `|Δweight| * capital`，若启用 `apply_fees`，使用真实交易金额调用 `compute_hk_fees()`，并将费用折算为对总资本的负收益：`fee_ret = - total_fee / capital`。
4. 将费用收益与市场收益相加形成 `net_returns`；累乘得到 `equity`。
5. 汇总交易列表与指标。

手续费简化假设：

- 仓位变动立刻成交，使用当日收盘价。
- 未考虑部分减仓、滑点、真实股数换算。
- 阶梯式平台费基于“当月已完成订单数 + 1”。

返回：`StrategyResult`

---

## 指标计算

### `compute_metrics(returns: pd.Series, equity: pd.Series, trades: pd.DataFrame)`

计算：

- `total_return`: 最终净值 - 1
- `cagr`: 按自然日 -> 年化复合增长率
- `max_drawdown`: 基于滚动最高点的回撤最小值
- `sharpe`: 日收益均值 / 标准差 * sqrt(252)
- `win_rate`: 非零仓位日中正收益的比例
- `trade_count`: 交易事件数量（BUY+SELL）
- `avg_trade_return`: 已完成 SELL 交易的平均单笔收益

注意：Sharpe 未扣除无风险利率；胜率按收益正负统计，不区分方向。

---

## 港股费用计算

### `_tiered_platform_fee(order_count_this_month: int)`

根据阶梯区间返回平台费（港元）。

### `compute_hk_fees(amount, order_count_this_month, instrument_type='stock', date=None, package='fixed', commission_free=False)`

返回费用字典：

- `commission`: 0.03% 成交额，最低3（可免佣）
- `platform_fee`: 固定15或阶梯式 (30→15→10→…→1)
- `system_fee`: 成交0.5港元（2023-01-01后取消）
- `settlement_fee`: 0.0042%
- `stamp_duty`: 股票 0.1% 向上取整最低1；ETF/窝轮/牛熊证免征
- `trading_fee`: 0.00565% 最低0.01
- `sfc_levy`: 0.0027% 最低0.01
- `frc_levy`: 0.00015%
- `total_fee`: 汇总之和

### `compute_negative_cash_interest(negative_cash: float, days: int, annual_rate=0.068)`

若账户现金为负，计算对应期间单利利息（当前未与回测净值结合）。

---

## CLI 参数

入口：`parse_args()` / `main()` 提供命令行操作。

主要参数：

- `--file`: 数据文件路径
- `--start / --end`: 日期过滤
- `--strategies`: 策略名称列表（来自 `STRATEGY_MAP`）
- `--output`: 汇总结果 CSV
- `--apply-fees`: 启用港股所有费用扣减
- `--capital`: 名义资金基数（决定费用影响）
- `--fee-package`: `fixed` | `tiered`
- `--instrument-type`: `stock|etf|warrant|cbbc|other` 影响印花税
- `--commission-free`: 免佣期（佣金设 0）
- `--max-position`: 单策略绝对持仓上限（0~1，可用于部分仓位控制）

示例：

```bash
python backtest_framework.py --file data/data_with_indicators.txt --start 2019-01-01 --end 2024-12-31 --strategies ma_cross rsi_reversion boll_breakout --apply-fees --capital 100000 --fee-package tiered --max-position 0.8 --output with_fees.csv
```

---

## 扩展策略示例

```python
class MyStrategy(Strategy):
    name = "my_strategy"
    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        # 简单示例：连续两日上涨则持仓
        sig = (close.pct_change() > 0) & (close.pct_change(2) > 0)
        return sig.astype(int)

# 注册
from backtest_framework import STRATEGY_MAP
STRATEGY_MAP['my_strategy'] = MyStrategy
```

---

## 设计局限与改进建议

- 开仓/平仓使用收盘价，未模拟开盘执行时的滑点与价格跳空。
- 手续费按名义资金计算，不基于真实股数与价格四舍五入规则。
- 权重视为理想值，不进行真实股数换算或余量跟踪。
- 不支持做空与保证金利息细化。
- 未记录费用明细表，可扩展返回费用时间序列以支持分解分析。

改进方向：

1. 引入成交价模型：`price_next_open` + 滑点模型（比例 / 绝对价）。
2. 部分仓位支持：`positions` 返回浮点仓位（0~1）。
3. 真实股数计算：股数 = 资金 / 价格，下单费用按实际交易金额精确计算。
4. 手续费明细输出：在 `trades` 中增加每笔费用列，或额外 `fees_df`。
5. 做空交易：扩展 positions = {-1,0,1} 并增加借券费用。
6. 多策略组合：添加组合加权与风险控制（最大回撤止损、波动目标）。

---

## 快速核查清单

- 是否启用手续费：检查命令行是否含 `--apply-fees`
- 平台套餐是否正确：`--fee-package fixed|tiered`
- 数据列是否包含所需指标：均线/RSI/BOLL 可直接复用或由策略内部计算
- 指标是否异常：极端 max_drawdown 或 sharpe 可能来自数据缺失或错误的持仓生成

---

## 免责声明

本框架仅用于学习研究，不构成真实投资建议；费用与税率以实际券商/监管披露为准。
