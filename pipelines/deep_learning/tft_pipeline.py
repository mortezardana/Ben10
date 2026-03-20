"""Simplified Temporal Fusion Transformer pipeline for gold trading."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from shared.interfaces import TradingPipeline, PipelineOutput
from utils.sequence_utils import create_sequences
from utils.logger import get_logger

logger = get_logger("tft_pipeline")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GatedResidualNetwork(nn.Module):
    """GRN block used in TFT."""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        h = torch.elu(self.fc1(x))
        h = self.dropout(h)
        output = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.layer_norm(skip + gate * output)


class SimplifiedTFT(nn.Module):
    """Simplified TFT: Variable Selection + GRN + Multi-Head Attention."""
    def __init__(self, n_features, hidden_size=64, n_heads=4, dropout=0.1, seq_len=32):
        super().__init__()
        self.hidden_size = hidden_size

        # Variable selection
        self.var_selection = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Softmax(dim=-1),
        )
        self.var_transform = nn.Linear(n_features, hidden_size)

        # Temporal processing (LSTM)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # GRN + output
        self.grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Variable selection
        weights = self.var_selection(x)  # (batch, seq, hidden)
        selected = self.var_transform(x) * weights

        # Temporal
        lstm_out, _ = self.lstm(selected)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)

        # Take last timestep
        final = attn_out[:, -1, :]

        # GRN + output
        final = self.grn(final)
        return torch.sigmoid(self.output(final))


class TFTPipeline(TradingPipeline):
    """Temporal Fusion Transformer pipeline."""

    def __init__(self, config=None):
        self.config = config or {
            'hidden_size': 64, 'n_heads': 4, 'dropout': 0.1,
            'learning_rate': 0.001, 'epochs': 50, 'batch_size': 64,
            'sequence_length': 32,
        }
        self.model = None
        self.feature_cols = None
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'

    @property
    def name(self): return "tft"

    def train(self, train_data, val_data=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TFT")

        self.feature_cols = [c for c in train_data.columns
                           if c not in ['target', 'date', 'Date', 'prediction']]
        self.feature_cols = train_data[self.feature_cols].select_dtypes(include='number').columns.tolist()

        seq_len = self.config.get('sequence_length', 32)
        X_raw = train_data[self.feature_cols].values
        y_raw = train_data['target'].values
        X_seq, y_seq = create_sequences(X_raw, y_raw, seq_len)

        n_features = X_seq.shape[2]
        self.model = SimplifiedTFT(
            n_features=n_features,
            hidden_size=self.config.get('hidden_size', 64),
            n_heads=self.config.get('n_heads', 4),
            dropout=self.config.get('dropout', 0.1),
            seq_len=seq_len,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.001))
        criterion = nn.BCELoss()

        X_t = torch.FloatTensor(X_seq).to(self.device)
        y_t = torch.FloatTensor(y_seq).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.get('batch_size', 64), shuffle=False)

        self.model.train()
        best_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(self.config.get('epochs', 50)):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb).squeeze()
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        preds = (self.model(X_t).squeeze() > 0.5).cpu().detach().numpy()
        acc = float((preds == y_seq).mean())
        logger.info(f"TFT trained: accuracy={acc:.4f}, epochs={epoch+1}")
        return {'accuracy': acc, 'loss': best_loss}

    def predict(self, data) -> PipelineOutput:
        self.model.eval()
        seq_len = self.config.get('sequence_length', 32)
        X_raw = data[self.feature_cols].values
        y_raw = data['target'].values if 'target' in data.columns else np.zeros(len(data))
        X_seq, _ = create_sequences(X_raw, y_raw, seq_len)

        with torch.no_grad():
            X_t = torch.FloatTensor(X_seq).to(self.device)
            probs = self.model(X_t).squeeze().cpu().numpy()

        idx = data.index[seq_len:]
        signals = pd.Series(np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)), index=idx)
        confidence = pd.Series(np.abs(probs - 0.5) * 2, index=idx)
        return PipelineOutput(signals=signals, confidence=confidence, metadata={'model': 'tft'})

    def evaluate(self, data):
        output = self.predict(data)
        seq_len = self.config.get('sequence_length', 32)
        y = data['target'].values[seq_len:]
        preds = (output.confidence > 0.5).astype(int).values
        acc = float((preds == y[:len(preds)]).mean()) if len(y) > 0 else 0
        return {'accuracy': acc}

    def save(self, path):
        torch.save({'state_dict': self.model.state_dict(), 'config': self.config, 'feature_cols': self.feature_cols}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.config = d['config']
        self.feature_cols = d['feature_cols']
        n_features = len(self.feature_cols)
        self.model = SimplifiedTFT(n_features=n_features, **{k: v for k, v in self.config.items() if k in ['hidden_size', 'n_heads', 'dropout', 'sequence_length']})
        self.model.load_state_dict(d['state_dict'])
        self.model.to(self.device)
