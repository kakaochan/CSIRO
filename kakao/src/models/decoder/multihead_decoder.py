import torch
import torch.nn as nn


class MultiHeadDecoder(nn.Module):
    """5つの独立したMLPヘッド"""
    def __init__(self, input_size, n_classes=5, dropout=0.3):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_size // 2, 1)
            ) for _ in range(n_classes)
        ])

    def forward(self, x):
        # x: (B, 1, n_features, 1) or (B, n_features)
        if x.dim() == 4:
            x = x.squeeze(1).squeeze(-1)  # (B, n_features)

        outputs = [head(x) for head in self.heads]  # 5 x (B, 1)
        return torch.cat(outputs, dim=1).unsqueeze(1)  # (B, 1, 5)
