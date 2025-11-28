import pandas as pd
import numpy as np
try:
    from .strategy_base import Strategy
except ImportError:
    from strategy_base import Strategy


class RandomForestStrategy(Strategy):
    name = "random_forest"

    def __init__(self):
        self.model = None
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20',
            'rsi14',
            'macd_dif', 'macd_dea', 'macd_bar',
            'boll_upper', 'boll_lower',
            'atr14'
        ]
        self.lags = 3  # RF might overfit with too many lags, keep it smaller
        self.cols_to_lag = ['close', 'volume', 'rsi14']
        self.full_df = None
        self.feature_names = []

    def prepare(self, data_path: str):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            print("Error: scikit-learn not installed. Please run 'pip install scikit-learn'.")
            return

        print(f"[{self.name}] Loading data for training from {data_path}...")
        df = pd.read_csv(data_path, sep='\t', parse_dates=['date'])
        df.set_index('date', inplace=True)
        self.full_df = df.copy()

        # Feature Engineering
        train_df = self._engineer_features(df.copy())
        train_df.dropna(inplace=True)

        # Target: Next day close > Today close
        train_df['target'] = (train_df['close'].shift(-1) > train_df['close']).astype(int)
        train_df.dropna(inplace=True)

        # Split for training
        train_data = train_df[train_df.index < '2024-01-01']
        
        if train_data.empty:
            print(f"[{self.name}] Warning: No training data found before 2024-01-01.")
            return

        X_train = train_data[self.feature_names]
        y_train = train_data['target']

        print(f"[{self.name}] Training Random Forest model on {len(X_train)} samples...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        print(f"[{self.name}] Training complete.")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_list = self.features.copy()
        
        # Lag features
        for col in self.cols_to_lag:
            if col not in df.columns: continue
            for lag in range(1, self.lags + 1):
                feat_name = f'{col}_lag_{lag}'
                df[feat_name] = df[col].shift(lag)
                feature_list.append(feat_name)

        # Return features
        df['return'] = df['close'].pct_change()
        for lag in range(1, self.lags + 1):
            feat_name = f'return_lag_{lag}'
            df[feat_name] = df['return'].shift(lag)
            feature_list.append(feat_name)
            
        self.feature_names = feature_list
        return df

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(0, index=df.index)

        if self.full_df is not None:
            full_feat = self._engineer_features(self.full_df.copy())
            valid_indices = df.index.intersection(full_feat.index)
            X_test = full_feat.loc[valid_indices, self.feature_names]
            
            # Predict probability
            probs = self.model.predict_proba(X_test)[:, 1]
            
            # Signal logic: > 0.55 (Slightly higher confidence for RF)
            signals = (probs > 0.55).astype(int)
            
            return pd.Series(signals, index=valid_indices).reindex(df.index).fillna(0)
        else:
            return pd.Series(0, index=df.index)

