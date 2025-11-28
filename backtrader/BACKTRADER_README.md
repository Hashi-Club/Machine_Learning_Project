# Backtrader 回测框架使用说明

本文档介绍了 `test_scripts/backtest_with_backtrader.py` 脚本的功能和使用方法。该脚本基于强大的 `backtrader` 库构建，旨在提供比简易回测框架更精确的交易模拟，特别针对港股市场进行了适配。

## 1. 功能特性

*   **策略复用**：直接兼容 `basic_strategies.py` 和 `ml_strategies.py` 中定义的策略逻辑，无需重写策略代码。
*   **港股仿真**：
    *   **完整手续费模型**：内置 `HKCommission` 类，精确计算佣金（0.03%）、平台费（15 HKD）、印花税（0.1%）、交易费、交易征费及结算费。
    *   **整手交易限制**：强制交易数量为 100 股的整数倍，符合港股交易规则。
*   **专业回测引擎**：利用 `backtrader` 的事件驱动引擎，处理订单执行、资金管理和资产估值。
*   **指标统计**：自动计算总收益率、年化收益率 (CAGR)、最大回撤、夏普比率、胜率等关键指标。

## 2. 环境依赖

运行此脚本需要安装 `backtrader` 库：

```bash
pip install backtrader
```

此外，脚本依赖项目中的其他标准库（`pandas`, `numpy` 等）。

## 3. 使用方法

脚本通过命令行参数进行配置。

### 命令行参数说明

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--file` | `data/data_with_indicators.txt` | 包含 OHLC 数据的源文件路径 |
| `--start` | `2024-01-01` | 回测起始日期 (YYYY-MM-DD) |
| `--end` | `2025-04-24` | 回测结束日期 (YYYY-MM-DD) |
| `--strategies` | (所有策略) | 指定要运行的策略名称列表（空格分隔） |
| `--output` | None | 将汇总结果保存为 CSV 文件的路径 |
| `--apply-fees` | False | **开关**：启用港股手续费扣除 |
| `--capital` | `100000.0` | 初始资金 |
| `--max-position` | `1.0` | 最大持仓权重（0.0 - 1.0），可用于控制仓位上限 |

### 支持的策略 (`--strategies`)

*   `buy_hold`: 买入并持有
*   `ma_cross`: 双均线交叉策略
*   `rsi_reversion`: RSI 均值回归策略
*   `boll_breakout`: 布林带突破策略
*   `macd`: MACD 趋势跟踪策略
*   `xgboost`: XGBoost 机器学习预测策略

## 4. 运行示例

### 示例 1：快速开始（运行所有策略，无手续费）
```bash
python test_scripts/backtest_with_backtrader.py
```

### 示例 2：启用港股手续费回测
```bash
python test_scripts/backtest_with_backtrader.py --apply-fees
```

### 示例 3：回测特定策略并指定日期范围
```bash
python test_scripts/backtest_with_backtrader.py --strategies macd xgboost --start 2023-01-01 --end 2023-12-31 --apply-fees
```

### 示例 4：保存结果到 CSV
```bash
python test_scripts/backtest_with_backtrader.py --output results.csv
```

## 5. 常见问题与注意事项

### 为什么开启手续费后，某些策略收益反而变高了？
这通常是因为**资金约束导致的被动择时**。
*   **现象**：在无手续费时，策略可能在某天满仓买入。开启手续费后，同样的资金因为不够支付手续费，导致第一笔买入失败（订单被拒）。
*   **结果**：策略被迫空仓等待，直到股价下跌，资金足以支付股票加手续费时才成交。如果这段时间股价下跌，这种“被迫等待”反而让策略以更低的价格建仓，从而获得了更高的收益。
*   **解决**：脚本已优化为**向下取整到 100 股**，通常能留出少量现金支付手续费，减少此类因资金不足导致的拒单情况。

### 关于整手交易
脚本强制将所有买卖数量向下取整为 100 的倍数。例如，如果资金理论上能买 356 股，脚本只会下单买入 300 股，剩余资金保留在账户中。

### 数据格式要求
输入的数据文件必须包含以下列（不区分大小写）：
*   `date`: 日期
*   `open`, `high`, `low`, `close`: 价格数据
*   `volume` (可选): 成交量（如果缺失默认为 0）
