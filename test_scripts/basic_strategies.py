import pandas as pd
import numpy as np
try:
    from .strategy_base import Strategy
except ImportError:
    from strategy_base import Strategy

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


class MACDStrategy(Strategy):
    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        # 优先使用数据中已有的 MACD 列 (假设参数匹配默认值 12, 26, 9)
        if all(c in df.columns for c in ['macd_dif', 'macd_dea']) and self.fast == 12 and self.slow == 26 and self.signal == 9:
            dif = df['macd_dif']
            dea = df['macd_dea']
        else:
            # 手动计算 MACD
            close = df['close']
            ema_fast = close.ewm(span=self.fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.slow, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=self.signal, adjust=False).mean()
        
        # 策略逻辑：DIF > DEA (金叉/多头区域) 持有，DIF < DEA (死叉/空头区域) 空仓
        pos = pd.Series(0, index=df.index)
        pos[dif > dea] = 1
        pos[dif < dea] = 0
        
        # 填充 NaN (通常是前几个数据点)
        return pos.fillna(0)
