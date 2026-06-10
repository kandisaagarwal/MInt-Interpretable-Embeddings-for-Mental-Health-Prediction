from __future__ import annotations

import torch
from torch import nn


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_classes: int = 7, # Number of output classes for classification
        pad_idx: int = 0,
        num_heads: int = 4,
        num_layers: int = 2, # TODO: Experiment with number of layers. Not too many though
        ff_dim: int = 256, # Hidden dimension of the ffn in the transformer
        dropout: float = 0.1,
        mlp_hidden_dim: int = 128, # Hidden dimension of the MLP classifier head
        # Sequence length mean: ~114, median: ~63, 95th percentile: ~399, 99th percentile: ~738
        max_len: int = 512, # Most inputs will fit in this context window
        freeze_embeddings: bool = False,
        max_pooling: bool = True, # If True, use max pooling instead of mean pooling
    ) -> None:
        super().__init__()

        if embedding_matrix.ndim != 2:
            raise ValueError("embedding_matrix must be 2D")

        vocab_size, embedding_dim = embedding_matrix.shape
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding dimension must be divisible by num_heads")

        self.max_pooling = max_pooling

        # pad_idx marks tokens added only for padding shorter sequences.
        self.pad_idx = pad_idx
        self.max_len = max_len

        # Token embedding table initialized from pretrained vectors.
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=pad_idx,
        )
        # Learn position embeddings to give more freedom to preserve interpretability
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Transformer stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu", # gelu is smoother than relu. can maybe help preserve interpretability of dimensions
            batch_first=True,
            norm_first=False, # post norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP classification head (pooled contextualized sequence -> 7 class logits)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' 
        input_ids: (batch size x sequence length) tensor of token IDs
        attention_mask: (batch size x sequence length) tensor where 1 indicates real tokens and 0 indicates padding
        Returns: logits (B, C), pooled sequence vector (B, D), token-level encoder outputs (B, L, D)
        '''
        
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")

        # Add positional embeddings to token embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = self.embedding(input_ids) + self.position_embedding(positions)
        # Dropout for regularization
        hidden_states = self.input_dropout(hidden_states)

        # Transformer expects True where positions should be ignored (padding).
        key_padding_mask = attention_mask == 0
        encoded = self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)

        # Masked mean pooling over valid tokens to get one vector per sequence.
        mask = attention_mask.unsqueeze(-1).type_as(encoded)
        # Use pooling to allow us to use interpretability term on contextualized embeddings
        if self.max_pooling:
            pooled = encoded.masked_fill(mask == 0, -1e9).max(dim=1).values
        else:
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # Return unnormalized class logits
        logits = self.classifier(pooled)

        # Pooled vector feeds the classifier; token outputs are used for interpretability regularization.
        return logits, pooled, encoded