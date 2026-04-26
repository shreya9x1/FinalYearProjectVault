"""
pointsam.py

A compact from-scratch Point-SAM style segmentation model implemented in PyTorch.

Model architecture (high level)
- Input: points [B, 3, N]
- PointEmbedding: per-point MLP -> embed_dim features [B, N, D]
- PositionalEmbedding: linear projection of xyz (optional) added to embeddings
- TransformerEncoder: several Transformer encoder layers (self-attention across N points)
- Decoder MLP: maps per-point features -> logits for num_classes

Outputs:
- logits: [B, N, num_classes]

Notes:
- Uses torch.nn.MultiheadAttention (operates on seq_len x batch x dim), so data is transposed accordingly.
- This implementation favors clarity and ease-of-use over special point-cloud ops (no FPS / ball query).
- Works well as a baseline and is straightforward to extend (add local grouping, hierarchical SA, etc.).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP block with optional BatchNorm and activation."""
    def __init__(self, channels, use_bn=True, activation=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i+1], bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(channels[i+1]))
            if i < len(channels) - 2:  # activation between layers, not after last
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        x = x.reshape(B * N, C)
        x = self.net(x)
        x = x.reshape(B, N, -1)
        return x


class PositionalEmbedding(nn.Module):
    """Simple learnable positional embedding from (x,y,z) coordinates."""
    def __init__(self, in_dim=3, out_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, coords):
        # coords: [B, 3, N] -> output [B, N, out_dim]
        coords = coords.permute(0, 2, 1).contiguous()  # [B, N, 3]
        return self.net(coords)


class SimpleTransformerEncoder(nn.Module):
    """A thin wrapper around PyTorch TransformerEncoder to operate on [B, N, C]."""
    def __init__(self, embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # we'll provide seq_len x batch x dim
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor] = None):
        # x: [B, N, C] -> Transformer expects [N, B, C]
        x = x.transpose(0, 1)  # [N, B, C]
        # src_key_padding_mask expects shape (B, N) if provided
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = out.transpose(0, 1)  # [B, N, C]
        return out


class PointSAM_Segmentation(nn.Module):
    """
    PointSAM-style segmentation network.

    Args:
        in_channels: input coordinate channels (3 normally)
        num_classes: number of part labels to predict per point
        embed_dim: internal embedding dim for point features
        transformer_layers: how many transformer encoder layers
        transformer_heads: number of attention heads
        mlp_hidden: hidden width multiplier for MLPs
        dropout: dropout applied to decoder
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 50,
        embed_dim: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        use_pos: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_pos = use_pos

        # Initial per-point embedding: map coords -> embed_dim
        # Accept [B, 3, N] -> produce [B, N, embed_dim]
        self.point_embed = MLP([in_channels, embed_dim // 2, embed_dim], use_bn=True)

        # Positional embedding (learned projection of coords)
        if use_pos:
            self.pos_embed = PositionalEmbedding(in_dim=in_channels, out_dim=embed_dim)

        # Transformer encoder for global context
        self.transformer = SimpleTransformerEncoder(
            embed_dim=embed_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=max(mlp_hidden, embed_dim * 2),
            dropout=dropout
        )

        # Decoder: per-point MLP -> logits
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, num_classes)
        )

        # small init
        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers with kaiming
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, points):
        """
        Forward pass.

        Args:
            points: [B, 3, N] or [B, 6, N] float tensor (xyz or xyz+rgb)
        Returns:
            logits: [B, N, num_classes] (no softmax)
        """
        assert points.dim() == 3, "points must be [B, C, N]"
        # If more than 3 channels, use only the first 3 (XYZ)
        if points.size(1) > 3:
            points = points[:, :3, :]
        assert points.size(1) == self.in_channels, f"points must be [B, {self.in_channels}, N]"

        B, C, N = points.shape

        # Convert to [B, N, C] for MLP and transformer
        pts = points.permute(0, 2, 1).contiguous()  # [B, N, 3]

        # Per-point embedding
        x = self.point_embed(pts)  # [B, N, embed_dim]

        # Positional embedding addition
        if self.use_pos:
            pos = self.pos_embed(points)  # [B, N, embed_dim]
            x = x + pos

        # Transformer expects [B, N, C] -> we handle inside wrapper
        x = self.transformer(x)  # [B, N, embed_dim]

        # Decoder: apply MLP per point. Decoder is built for BN working on (B*N, C)
        BxN, feat_dim = B * N, x.shape[-1]
        x_flat = x.reshape(BxN, feat_dim)  # [B*N, feat]

        logits_flat = self.decoder(x_flat)  # [B*N, num_classes]
        logits = logits_flat.reshape(B, N, self.num_classes)  # [B, N, num_classes]

        return logits


# -------------------------------
# Quick smoke test (run as script)

# -------------------------------
'''
if __name__ == "__main__":
    # small local test to verify shapes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    N = 1024
    num_classes = 50

    model = PointSAM_Segmentation(in_channels=3, num_classes=num_classes, embed_dim=128,
                                  transformer_layers=3, transformer_heads=4, mlp_hidden=256,
                                  dropout=0.1, use_pos=True).to(device)
    pts = torch.randn(B, 3, N, device=device)
    logits = model(pts)
    print("logits shape:", logits.shape)  # expect [B, N, num_classes]
    assert logits.shape == (B, N, num_classes)
    print("pointsam.py smoke test passed.")
'''