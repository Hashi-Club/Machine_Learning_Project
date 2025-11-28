import pandas as pd
import numpy as np
try:
    from .strategy_base import Strategy
except ImportError:
    from strategy_base import Strategy


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
        self.seq_len = 10  # LSTM 序列长度
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

        # 特征工程 (基础特征 + 归一化)
        # LSTM 处理序列数据，因此不需要手动创建滞后特征列，
        # 但需要对数据进行归一化处理。
        
        # 确保特征存在
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            pass # 假设特征已存在
        
        self.feature_names = self.features
        
        # 准备训练数据
        train_df = df[df.index < '2024-01-01'].copy()
        train_df.dropna(inplace=True)
        if train_df.empty:
            print(f"[{self.name}] Warning: No training data found before 2024-01-01.")
            return

        # 数据归一化
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_scaled = self.scaler.fit_transform(train_df[self.feature_names])
        
        # 创建时间序列数据
        X_train, y_train = self._create_sequences(train_data_scaled, train_df['close'].values, self.seq_len)
        
        if len(X_train) == 0:
            print(f"[{self.name}] Not enough data for sequence length {self.seq_len}")
            return

        # 转换为 PyTorch 张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        # 定义 LSTM 模型
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super(LSTMModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        input_dim = len(self.feature_names)
        hidden_dim = 128
        num_layers = 2
        output_dim = 1
        
        self.model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(self.device)
        
        # 训练循环
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        epochs = 300
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
        # 目标: 如果 Close[t+1] > Close[t] 则为 1，否则为 0
        # 我们需要序列 [t-seq_len+1 ... t] 来预测 t+1 的走势
        
        for i in range(len(data) - seq_len - 1):
            x = data[i:(i+seq_len)]
            # 目标: 比较 close[i+seq_len] (当前) 和 close[i+seq_len+1] (下一时刻)
            
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

        # 我们需要完整的上下文来生成序列
        if self.full_df is not None:
            # 准备完整数据
            # 使用训练好的 scaler 进行归一化
            # 注意: 在生产环境中，scaler 应该只在训练数据上拟合 (已在 prepare 中完成)
            
            # 我们需要为 df 中的索引生成预测
            # 但每个预测都需要 seq_len 的回溯窗口
            
            # 为简化起见，处理整个 full_df 然后切片
            # 对于大数据量可能较慢，但在此处可行
            
            full_data_scaled = self.scaler.transform(self.full_df[self.feature_names])
            
            # 生成推理用的序列
            # 我们想预测时间 t。输入是 [t-seq_len+1 ... t]
            # 因此对于 full_df 中的每一行 (从 seq_len 开始)，我们可以预测下一天
            
            signals = pd.Series(0, index=self.full_df.index)
            
            self.model.eval()
            with torch.no_grad():
                # 批量处理
                # 创建所有序列
                # 对于索引 t，我们需要 data[t-seq_len+1 : t+1]
                
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
                
                # 批量预测
                batch_size = 128
                all_preds = []
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    outputs = self.model(batch)
                    probs = torch.sigmoid(outputs)
                    all_preds.extend(probs.cpu().numpy().flatten())
                
                # 阈值判断
                pred_signals = (np.array(all_preds) > 0.52).astype(int)
                
                signals.loc[valid_indices] = pred_signals

            return signals.reindex(df.index).fillna(0)
        else:
            return pd.Series(0, index=df.index)