import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import mplfinance as mpf


def load_data(path: Path) -> pd.DataFrame:
    """Load tab-delimited OHLC data file into a DataFrame suitable for mplfinance.

    The file is expected to have at least the columns:
    date, open, high, low, close, volume
    """
    df = pd.read_csv(path, sep='\t', engine='python')
    # Normalize column names (lowercase) for consistency
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.set_index('date').sort_index()
    return df


def build_addplots(df: pd.DataFrame, enable: bool):
    """Build list of addplots for mplfinance.

    Price panel (panel=0): MA, EMA, Bollinger Bands overlay.
    Separate panels: RSI, MACD (lines + histogram), ATR.
    Panel numbering: 0=price, then increment for each extra indicator present.
    """
    if not enable:
        return []

    addplots = []

    # Moving Averages
    for ma_col in ['ma5', 'ma10', 'ma20']:
        if ma_col in df.columns and df[ma_col].notna().any():
            addplots.append(mpf.make_addplot(df[ma_col], ylabel='MA'))

    # EMAs overlay
    for ema_col in ['ema12', 'ema26']:
        if ema_col in df.columns and df[ema_col].notna().any():
            addplots.append(mpf.make_addplot(df[ema_col], ylabel='EMA'))

    # Bollinger Bands overlay (mid, upper, lower)
    boll_cols = ['boll_mid', 'boll_upper', 'boll_lower']
    if all(c in df.columns for c in boll_cols):
        for bc in boll_cols:
            if df[bc].notna().any():
                addplots.append(mpf.make_addplot(df[bc], ylabel='BOLL'))

    # Dynamic panel index begins at 1 (panel 0 is price)
    panel_index = 1

    # RSI
    if 'rsi14' in df.columns and df['rsi14'].notna().any():
        addplots.append(mpf.make_addplot(df['rsi14'], panel=panel_index, ylabel='RSI'))
        panel_index += 1

    # MACD: dif, dea, bar histogram
    macd_components = ['macd_dif', 'macd_dea', 'macd_bar']
    if any(c in df.columns for c in macd_components):
        # Ensure at least one valid series to avoid empty panel
        macd_series_present = [c for c in macd_components if c in df.columns and df[c].notna().any()]
        if macd_series_present:
            # Lines first (dif, dea)
            if 'macd_dif' in df.columns and df['macd_dif'].notna().any():
                addplots.append(mpf.make_addplot(df['macd_dif'], panel=panel_index, ylabel='MACD'))
            if 'macd_dea' in df.columns and df['macd_dea'].notna().any():
                addplots.append(mpf.make_addplot(df['macd_dea'], panel=panel_index))
            # Histogram (macd_bar)
            if 'macd_bar' in df.columns and df['macd_bar'].notna().any():
                addplots.append(mpf.make_addplot(df['macd_bar'], type='bar', panel=panel_index))
            panel_index += 1

    # ATR
    if 'atr14' in df.columns and df['atr14'].notna().any():
        addplots.append(mpf.make_addplot(df['atr14'], panel=panel_index, ylabel='ATR'))
        panel_index += 1

    return addplots


def plot(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, indicators: bool = False,
         save: Optional[Path] = None, style: str = 'yahoo'):
    # Date filtering
    if start:
        start_dt = pd.to_datetime(start, errors='coerce')
        df = df[df.index >= start_dt]
    if end:
        end_dt = pd.to_datetime(end, errors='coerce')
        df = df[df.index <= end_dt]

    if df.empty:
        raise ValueError('过滤后没有可绘制的数据')

    addplots = build_addplots(df, indicators)
    kwargs = dict(type='candle', style=style, volume=True, addplot=addplots, figratio=(16, 9), figscale=1.2)

    if save:
        mpf.plot(df, **kwargs, savefig=dict(fname=str(save), dpi=150, bbox_inches='tight'))
    else:
        mpf.plot(df, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='读取数据并绘制蜡烛图')
    parser.add_argument('--file', default='data/data_with_indicators.txt', help='数据文件路径')
    parser.add_argument('--start', help='起始日期 YYYY-MM-DD', default=None)
    parser.add_argument('--end', help='结束日期 YYYY-MM-DD', default=None)
    parser.add_argument('--indicators', action='store_true', help='叠加MA/EMA/BOLL以及RSI、MACD、ATR等指标')
    parser.add_argument('--save', help='保存为图片路径 (png/jpg)', default=None)
    parser.add_argument('--style', help='mplfinance样式 (yahoo, charles, binance, classic...)', default='yahoo')
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f'数据文件不存在: {path}')

    df = load_data(path)
    plot(df, start=args.start, end=args.end, indicators=args.indicators,
         save=Path(args.save) if args.save else None, style=args.style)


if __name__ == '__main__':
    main()
