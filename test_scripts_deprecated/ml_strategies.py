import pandas as pd
import numpy as np
try:
    from .strategy_base import Strategy
except ImportError:
    from strategy_base import Strategy

class XGBoostStrategy(Strategy):
    name = "xgboost"

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
        self.lags = 5
        self.cols_to_lag = ['close', 'volume', 'rsi14', 'macd_bar']
        self.full_df = None
        self.feature_names = []

    def prepare(self, data_path: str):
        try:
            import xgboost as xgb
        except ImportError:
            print("Error: xgboost not installed. Please run 'pip install xgboost'.")
            return

        print(f"[{self.name}] Loading data for training from {data_path}...")
        df = pd.read_csv(data_path, sep='\t', parse_dates=['date'])
        df.set_index('date', inplace=True)
        self.full_df = df.copy() # Keep full data for feature engineering context

        # Feature Engineering
        train_df = self._engineer_features(df.copy())
        train_df.dropna(inplace=True)

        # Target: Next day close > Today close
        train_df['target'] = (train_df['close'].shift(-1) > train_df['close']).astype(int)
        train_df.dropna(inplace=True)

        # Split for training (Fixed split as per 1.py logic)
        # Train on 2018-2023
        train_data = train_df[train_df.index < '2024-01-01']
        
        if train_data.empty:
            print(f"[{self.name}] Warning: No training data found before 2024-01-01.")
            return

        X_train = train_data[self.feature_names]
        y_train = train_data['target']

        print(f"[{self.name}] Training XGBoost model on {len(X_train)} samples...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)
        print(f"[{self.name}] Training complete.")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Base features must exist
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            # Try to compute missing indicators if possible, or warn
            # For now assume they exist as per data_with_indicators.txt
            pass
        
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

        # We need context for lag features. 
        # If self.full_df is available, use it to generate features for the requested dates.
        if self.full_df is not None:
            # Use full_df to generate features, then slice
            full_feat = self._engineer_features(self.full_df.copy())
            # Align with requested df
            # We need to predict for the dates in df
            # But be careful: we need features at time T to predict T+1 return (signal for T)
            
            # Filter for the relevant dates
            # We need to ensure we have data for the requested index
            valid_indices = df.index.intersection(full_feat.index)
            X_test = full_feat.loc[valid_indices, self.feature_names]
            
            # Predict
            # predict_proba returns [prob_0, prob_1]
            probs = self.model.predict_proba(X_test)[:, 1]
            
            # Signal logic: > 0.6 -> 1 (Buy/Hold), else 0
            # Increased threshold from 0.55 to 0.6 to account for transaction fees
            signals = (probs > 0.5).astype(int)
            
            return pd.Series(signals, index=valid_indices).reindex(df.index).fillna(0)
        else:
            # Fallback if prepare wasn't called (shouldn't happen if framework is updated)
            return pd.Series(0, index=df.index)


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


class SVMStrategy(Strategy):
    name = "svm"

    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20',
            'rsi14',
            'macd_dif', 'macd_dea', 'macd_bar',
            'boll_upper', 'boll_lower',
            'atr14'
        ]
        self.lags = 3
        self.cols_to_lag = ['close', 'volume', 'rsi14']
        self.full_df = None
        self.feature_names = []

    def prepare(self, data_path: str):
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
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

        # Scaling is crucial for SVM
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        print(f"[{self.name}] Training SVM model on {len(X_train)} samples...")
        # Using RBF kernel, probability=True for predict_proba
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
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
        if self.model is None or self.scaler is None:
            return pd.Series(0, index=df.index)

        if self.full_df is not None:
            full_feat = self._engineer_features(self.full_df.copy())
            valid_indices = df.index.intersection(full_feat.index)
            X_test = full_feat.loc[valid_indices, self.feature_names]
            
            # Scale features using the same scaler fitted on training data
            X_test_scaled = self.scaler.transform(X_test)
            
            # Predict probability
            probs = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Signal logic: > 0.55
            signals = (probs > 0.55).astype(int)
            
            return pd.Series(signals, index=valid_indices).reindex(df.index).fillna(0)
        else:
            return pd.Series(0, index=df.index)


class LSTMStrategy(Strategy):
    name = "lstm"

    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20',
            'rsi14',
            'macd_dif', 'macd_dea', 'macd_bar',
            'boll_upper', 'boll_lower',
            'atr14'
        ]
        self.seq_len = 10  # Sequence length for LSTM
        self.full_df = None
        self.feature_names = []
        self.device = None

    def prepare(self, data_path: str):
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import MinMaxScaler
        except ImportError:
            print("Error: torch or scikit-learn not installed. Please run 'pip install torch scikit-learn'.")
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[{self.name}] Using device: {self.device}")

        print(f"[{self.name}] Loading data for training from {data_path}...")
        df = pd.read_csv(data_path, sep='\t', parse_dates=['date'])
        df.set_index('date', inplace=True)
        self.full_df = df.copy()

        # Feature Engineering (Basic + Scaling)
        # LSTM handles sequences, so we don't need manual lag features as columns,
        # but we need to scale the data.
        
        # Ensure features exist
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            pass # Assume exists
        
        self.feature_names = self.features
        
        # Prepare training data
        train_df = df[df.index < '2024-01-01'].copy()
        if train_df.empty:
            print(f"[{self.name}] Warning: No training data found before 2024-01-01.")
            return

        # Scale data
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_scaled = self.scaler.fit_transform(train_df[self.feature_names])
        
        # Create sequences
        X_train, y_train = self._create_sequences(train_data_scaled, train_df['close'].values, self.seq_len)
        
        if len(X_train) == 0:
            print(f"[{self.name}] Not enough data for sequence length {self.seq_len}")
            return

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        # Define Model
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super(LSTMModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return self.sigmoid(out)

        input_dim = len(self.feature_names)
        hidden_dim = 64
        num_layers = 2
        output_dim = 1
        
        self.model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(self.device)
        
        # Training Loop
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        epochs = 50
        batch_size = 32
        
        print(f"[{self.name}] Training LSTM model on {len(X_train)} samples for {epochs} epochs...")
        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(X_train_tensor.size()[0])
            epoch_loss = 0
            for i in range(0, X_train_tensor.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train_tensor):.6f}")
                
        print(f"[{self.name}] Training complete.")

    def _create_sequences(self, data, close_prices, seq_len):
        xs, ys = [], []
        # Target: 1 if Close[t+1] > Close[t], else 0
        # We need sequence [t-seq_len+1 ... t] to predict t+1 movement
        # Actually, standard is: use [t-seq_len ... t-1] to predict t
        # Here we want to predict direction for t+1 based on data up to t
        
        for i in range(len(data) - seq_len - 1):
            x = data[i:(i+seq_len)]
            # Target: Compare close[i+seq_len] (current) with close[i+seq_len+1] (next)
            # Wait, if we are at time t (index i+seq_len-1 in data window), we want to predict t+1
            # So target is (close[i+seq_len] > close[i+seq_len-1]) ?
            
            # Let's align:
            # x: data[i : i+seq_len] -> This is data from t-seq_len to t-1
            # We want to predict movement at t (relative to t-1)
            # So target is close[i+seq_len] > close[i+seq_len-1]
            
            # But for trading, we usually want: At time t, predict t+1
            # So input is data[i : i+seq_len] (data up to t)
            # Target is close[i+seq_len+1] > close[i+seq_len]
            
            current_close = close_prices[i+seq_len-1]
            next_close = close_prices[i+seq_len]
            
            target = 1 if next_close > current_close else 0
            
            xs.append(x)
            ys.append(target)
            
        return np.array(xs), np.array(ys)

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            return pd.Series(0, index=df.index)
            
        import torch

        # We need full context to generate sequences
        if self.full_df is not None:
            # Prepare full data
            # We need to scale using the fitted scaler
            # Note: In production, we should fit scaler only on training data (done in prepare)
            
            # We need to generate predictions for the indices in df
            # But each prediction needs seq_len lookback
            
            # Let's get the relevant slice from full_df including lookback
            # Find start index of df in full_df
            
            # To simplify, let's process the whole full_df and then slice
            # This might be slow for huge data but fine here
            
            full_data_scaled = self.scaler.transform(self.full_df[self.feature_names])
            
            # Generate sequences for inference
            # We want to predict for time t. Input is [t-seq_len+1 ... t]
            # So for each row in full_df (starting from seq_len), we can make a prediction for the NEXT day
            # But wait, our target in training was: input [0..9] -> predict movement 9->10?
            # In _create_sequences:
            # x = data[i : i+seq_len] (indices 0..9)
            # target based on close[i+seq_len] (index 10) vs close[i+seq_len-1] (index 9)
            # So input [0..9] predicts direction of 10.
            
            # So at index t (time t), we want to predict t+1.
            # We need input [t-seq_len+1 ... t]
            
            signals = pd.Series(0, index=self.full_df.index)
            
            self.model.eval()
            with torch.no_grad():
                # We can batch this
                # Create all sequences
                # For index t, we need data[t-seq_len+1 : t+1]
                
                # Let's iterate only for the needed range to save time?
                # Or just iterate all valid
                
                X_all = []
                valid_indices = []
                
                data_values = full_data_scaled
                
                for i in range(self.seq_len, len(data_values)):
                    # Sequence ending at i (inclusive) -> data[i-seq_len+1 : i+1]
                    # Length is seq_len
                    seq = data_values[i-self.seq_len+1 : i+1]
                    X_all.append(seq)
                    valid_indices.append(self.full_df.index[i])
                
                if not X_all:
                    return pd.Series(0, index=df.index)
                
                X_tensor = torch.FloatTensor(np.array(X_all)).to(self.device)
                
                # Predict in batches
                batch_size = 128
                all_preds = []
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    outputs = self.model(batch)
                    all_preds.extend(outputs.cpu().numpy().flatten())
                
                # Threshold
                pred_signals = (np.array(all_preds) > 0.55).astype(int)
                
                # Map back to dates
                # valid_indices[k] corresponds to prediction made at time valid_indices[k] for NEXT day?
                # No, our training logic:
                # x = data[i : i+seq_len] (0..9) -> target (10 vs 9)
                # So input ending at index 9 predicts movement at index 10.
                # So at time t (index 9), we predict t+1.
                # So the signal calculated using data up to t should be assigned to t.
                
                signals.loc[valid_indices] = pred_signals

            return signals.reindex(df.index).fillna(0)
        else:
            return pd.Series(0, index=df.index)


