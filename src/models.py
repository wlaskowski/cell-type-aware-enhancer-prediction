"""
Define all neural network architectures
"""

import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """
    Learn how strongly each sequence position should contribute
    to the final sequence representation.
    """

    def __init__(self, channels):
        # convert the feature vector at every sequence position into attention score
        super().__init__()
        self.attention = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: torch tensor with shape: (batch, channels, len)
        # returns weighted sequence representation with shape: (batch, channels)
        
        # produce one raw attention score for every sequence position
        scores = self.attention(x)

        # convert raw scores into normalized attention weights
        weights = torch.softmax(scores, dim=-1)
        
        # 
        pooled = torch.sum(x * weights, dim=-1)
        return pooled
    

class BaselineCNN(nn.Module):
    """
    Sequence-only convolutional neural network. 
    Predicts regulatory activity using only the one-hot encoded DNA.
    """
    def __init__(self):
        super().__init__()

        # extract local sequence patterns, conv layers
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

        # average every feature channel across all sequence positions
        self.pool = nn.AdaptiveAvgPool1d(1)

        # converts features into activity predictions
        self.regressor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
        )

    def forward(self, x_seq, x_cell=None):
        # x_cell is accepted for compatibility with other model architectures
        seq_features = self.sequence_encoder(x_seq)
        seq_features = self.pool(seq_features).squeeze(-1)
        return self.regressor(seq_features).squeeze(-1)
    

class CellEmbeddingCNN(BaselineCNN):
    """
    CNN that combines DNA sequence features with a cell-type embeddings
    """
    def __init__(self, n_cell_types=3, cell_embedding_dim=6):
        # inhariting encoder and pooling from baseline
        super().__init__()

        # create one trainable six-dim vector 
        self.cell_embedding = nn.Embedding(n_cell_types, cell_embedding_dim)

        # replacing baseline regression head (adding 6 cell-type features)
        self.regressor = nn.Sequential(
            nn.Linear(16 + cell_embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
        )

    def forward(self, x_seq, x_cell):
    
        seq_features = self.sequence_encoder(x_seq)
        seq_features = self.pool(seq_features).squeeze(-1)
        
        # retrieving learned embedding vector for every cell type
        cell_features = self.cell_embedding(x_cell)

        # combining both representations (sequence features and cell-type information)
        combined = torch.cat([seq_features, cell_features], dim=-1)
        return self.regressor(combined).squeeze(-1)
    

class AttentionCNN(BaselineCNN):
    """
    Sequence-only CNN using attention-based pooling instead of global average
    """
    def __init__(self):
        # initialize from baseline
        super().__init__()

        # replace pooling with learned attention pools
        self.pool = AttentionPooling(channels=16)

    def forward(self, x_seq, x_cell=None):

        seq_features = self.sequence_encoder(x_seq)
        
        # attention mechanism 
        seq_features = self.pool(seq_features)
        return self.regressor(seq_features).squeeze(-1)
    

class FullCNN(CellEmbeddingCNN):
    """
    CNN combining cell-type embeddings with attention-based pooling
    """
    def __init__(self, n_cell_types=3, cell_embedding_dim=6):
        super().__init__(n_cell_types, cell_embedding_dim)

         # again attention pooling
        self.pool = AttentionPooling(channels=16)

    def forward(self, x_seq, x_cell):
        seq_features = self.sequence_encoder(x_seq)
       
        seq_features = self.pool(seq_features)

        cell_features = self.cell_embedding(x_cell)

        combined = torch.cat([seq_features, cell_features], dim=-1)
        return self.regressor(combined).squeeze(-1)
    

def get_model(variant):
    """
    Function creates the model selected through command-line 
    """
    if variant == "baseline":
        return BaselineCNN()

    if variant == "embedding":
        return CellEmbeddingCNN()

    if variant == "attention":
        return AttentionCNN()

    if variant == "full":
        return FullCNN()

    raise ValueError(f"Unknown model variant: {variant}")