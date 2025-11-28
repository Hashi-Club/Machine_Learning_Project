from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd

@dataclass
class StrategyResult:
    name: str
    metrics: Dict[str, float]
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame

class Strategy:
    name: str = "Base"

    def prepare(self, data_path: str):
        """可选：在回测前进行准备工作（如训练模型）。"""
        pass

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """返回每日持仓权重 ([-1, 1] 浮点)，index 与 df 一致。"""
        raise NotImplementedError
