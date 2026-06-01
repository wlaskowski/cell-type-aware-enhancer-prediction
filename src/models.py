import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, channels, len)
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        return pooled
    

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2),

            nn.Conv1d(8, 8, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(8),

            nn.Conv1d(8, 16, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
        )

    def forward(self, x_seq, x_cell=None):
        seq_features = self.sequence_encoder(x_seq)
        seq_features = self.pool(seq_features).squeeze(-1)
        return self.regressor(seq_features).squeeze(-1)
    

class CellEmbeddingCNN(BaselineCNN):
    def __init__(self, n_cell_types=3, cell_embedding_dim=6):
        super().__init__()

        self.cell_embedding = nn.Embedding(n_cell_types, cell_embedding_dim)

        self.regressor = nn.Sequential(
            nn.Linear(16 + cell_embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
        )

    def forward(self, x_seq, x_cell):
        seq_features = self.sequence_encoder(x_seq)
        seq_features = self.pool(seq_features).squeeze(-1)

        cell_features = self.cell_embedding(x_cell)

        combined = torch.cat([seq_features, cell_features], dim=-1)
        return self.regressor(combined).squeeze(-1)
    

class AttentionCNN(BaselineCNN):
    def __init__(self):
        super().__init__()

        self.pool = AttentionPooling(channels=16)

    def forward(self, x_seq, x_cell=None):
        seq_features = self.sequence_encoder(x_seq)
        seq_features = self.pool(seq_features)
        return self.regressor(seq_features).squeeze(-1)
    

class FullCNN(CellEmbeddingCNN):
    def __init__(self, n_cell_types=3, cell_embedding_dim=6):
        super().__init__(n_cell_types, cell_embedding_dim)

        self.pool = AttentionPooling(channels=16)

    def forward(self, x_seq, x_cell):
        seq_features = self.sequence_encoder(x_seq)
        seq_features = self.pool(seq_features)

        cell_features = self.cell_embedding(x_cell)

        combined = torch.cat([seq_features, cell_features], dim=-1)
        return self.regressor(combined).squeeze(-1)
    

def get_model(variant):
    if variant == "baseline":
        return BaselineCNN()

    if variant == "embedding":
        return CellEmbeddingCNN()

    if variant == "attention":
        return AttentionCNN()

    if variant == "full":
        return FullCNN()

    raise ValueError(f"Unknown model variant: {variant}")