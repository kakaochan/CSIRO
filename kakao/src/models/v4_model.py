"""V4Model - CrossPVT_T2T_MambaDINO Architecture

V4モデル - CrossPVT_T2T_MambaDINOアーキテクチャ

This module implements the V4 (CrossPVT_T2T_MambaDINO) architecture from REFERENCE.
It combines multiple advanced techniques:
- DINO backbone (Vision Transformer with self-supervised learning)
- TileEncoder (multi-grid tile processing: 4x4 small, 2x2 big)
- T2T (Tokens-to-Token retokenization)
- CrossScaleFusion (cross-attention between different scales)
- PyramidMixer (MobileViT → PVT → Mamba stages)
- Gating mechanism (left/right mutual modulation)

このモジュールはREFERENCEからV4（CrossPVT_T2T_MambaDINO）アーキテクチャを実装します。
複数の先進技術を組み合わせています：
- DINOバックボーン（自己教師あり学習を用いたVision Transformer）
- TileEncoder（マルチグリッドタイル処理：4x4小、2x2大）
- T2T（Tokens-to-Tokenリトークン化）
- CrossScaleFusion（異なるスケール間のクロスアテンション）
- PyramidMixer（MobileViT → PVT → Mambaステージ）
- ゲーティングメカニズム（左右相互変調）
"""

import math
from typing import Optional, Tuple, Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# =============================================================================
# Basic Building Blocks / 基本構成ブロック
# =============================================================================

class FeedForward(nn.Module):
    """MLP with GELU activation / GELU活性化関数を使用したMLP"""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    """Standard Transformer block with multi-head attention + feed-forward

    マルチヘッドアテンション + フィードフォワードを持つ標準Transformerブロック
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual / 残差接続付き自己アテンション
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # Feed-forward with residual / 残差接続付きフィードフォワード
        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# MobileViT Block / MobileViTブロック
# =============================================================================

class MobileViTBlock(nn.Module):
    """Lightweight MobileViT: Local CNN + Small Transformer

    軽量MobileViT：局所CNN + 小型Transformer

    Combines local feature extraction (CNN) with global modeling (Transformer).
    局所特徴抽出（CNN）とグローバルモデリング（Transformer）を組み合わせます。
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        depth: int = 2,
        patch: Tuple[int, int] = (2, 2),
        dropout: float = 0.0,
    ):
        super().__init__()
        # Local feature extraction / 局所特徴抽出
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # Depthwise conv
            nn.Conv2d(dim, dim, 1),  # Pointwise conv
            nn.GELU(),
        )
        self.patch = patch

        # Transformer layers / Transformerレイヤー
        self.transformer = nn.ModuleList([
            AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0)
            for _ in range(depth)
        ])

        # Fusion layer / 融合レイヤー
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map / 特徴マップ
        Returns:
            (B, C, H, W) fused features / 融合された特徴
        """
        # Extract local features / 局所特徴抽出
        local_feat = self.local(x)
        B, C, H, W = local_feat.shape

        # Ensure dimensions are divisible by patch size / パッチサイズで割り切れるよう調整
        ph, pw = self.patch
        new_h = math.ceil(H / ph) * ph
        new_w = math.ceil(W / pw) * pw
        if new_h != H or new_w != W:
            local_feat = F.interpolate(
                local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            H, W = new_h, new_w

        # Unfold to patches and create tokens / パッチに展開してトークン化
        tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, C, nh, nw, ph, pw)
        tokens = tokens.contiguous().view(B, C, -1, ph, pw)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # (B, N, C)

        # Apply transformer / Transformerを適用
        for blk in self.transformer:
            tokens = blk(tokens)

        # Fold back to feature map / 特徴マップに戻す
        feat = tokens.view(B, -1, ph * pw, C).permute(0, 3, 1, 2)
        nh = H // ph
        nw = W // pw
        feat = feat.view(B, C, nh, nw, ph, pw).permute(0, 1, 2, 4, 3, 5)
        feat = feat.reshape(B, C, H, W)

        # Resize to match input / 入力サイズに合わせる
        if feat.shape[-2:] != x.shape[-2:]:
            feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Fuse local and global features / 局所特徴とグローバル特徴を融合
        out = self.fuse(torch.cat([x, feat], dim=1))
        return out


# =============================================================================
# PVT (Pyramid Vision Transformer) Block / PVT（ピラミッドビジョントランスフォーマー）ブロック
# =============================================================================

class SpatialReductionAttention(nn.Module):
    """Spatial Reduction Attention from PVT

    PVTの空間削減アテンション

    Uses spatial reduction to reduce computational complexity.
    空間削減を使用して計算量を削減します。
    """

    def __init__(self, dim: int, heads: int = 8, sr_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None

        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) token sequence / トークンシーケンス
            hw: (H, W) spatial dimensions / 空間次元
        Returns:
            (B, N, C) attended features / アテンション後の特徴
        """
        B, N, C = x.shape

        # Compute query / クエリを計算
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        # Compute key-value with spatial reduction / 空間削減付きでkey-valueを計算
        if self.sr is not None:
            H, W = hw
            feat = x.transpose(1, 2).reshape(B, C, H, W)
            feat = self.sr(feat)  # Spatial reduction / 空間削減
            feat = feat.reshape(B, C, -1).transpose(1, 2)
            feat = self.norm(feat)
        else:
            feat = x

        kv = self.kv(feat)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 3, 1)
        v = v.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)

        # Attention / アテンション
        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class PVTBlock(nn.Module):
    """PVT block with spatial reduction attention

    空間削減アテンションを持つPVTブロック
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        sr_ratio: int = 2,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.sra = SpatialReductionAttention(dim, heads=heads, sr_ratio=sr_ratio, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) token sequence / トークンシーケンス
            hw: (H, W) spatial dimensions / 空間次元
        Returns:
            (B, N, C) processed tokens / 処理されたトークン
        """
        x = x + self.sra(self.norm1(x), hw)
        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# Mamba Block / Mambaブロック
# =============================================================================

class LocalMambaBlock(nn.Module):
    """Simplified local Mamba: DW-Conv + gating + linear projection

    簡略化された局所Mamba：DW-Conv + ゲーティング + 線形射影

    Mamba is a state space model for sequential dependencies.
    Mambaは系列依存性のための状態空間モデルです。
    """

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) token sequence / トークンシーケンス
        Returns:
            (B, N, C) processed tokens / 処理されたトークン
        """
        shortcut = x
        x = self.norm(x)

        # Gating mechanism / ゲーティングメカニズム
        g = torch.sigmoid(self.gate(x))
        x = (x * g).transpose(1, 2)  # (B, C, N)

        # Depthwise convolution / 深さ方向畳み込み
        x = self.dwconv(x).transpose(1, 2)  # (B, N, C)

        # Projection and residual / 射影と残差接続
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


# =============================================================================
# T2T (Tokens-to-Token) Retokenizer / T2T（トークン間）リトークナイザー
# =============================================================================

class T2TRetokenizer(nn.Module):
    """Tokens-to-Token retokenization

    トークン間リトークン化

    Compresses 4x4 tile tokens to 2x2 through local attention.
    局所アテンションを通じて4x4タイルトークンを2x2に圧縮します。
    """

    def __init__(self, dim: int, depth: int = 2, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0)
            for _ in range(depth)
        ])

    def forward(
        self, tokens: torch.Tensor, grid_hw: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, T, C) tile tokens / タイルトークン
            grid_hw: (H, W) grid dimensions / グリッド次元
        Returns:
            retokens: (B, 4, C) compressed tokens (2x2 grid) / 圧縮されたトークン（2x2グリッド）
            seq_map: (B, C, H, W) intermediate feature map / 中間特徴マップ
        """
        B, T, C = tokens.shape
        H, W = grid_hw

        # Reshape to feature map / 特徴マップに変形
        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
        seq = feat_map.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Apply attention blocks / アテンションブロックを適用
        for blk in self.blocks:
            seq = blk(seq)

        # Reshape back and pool to 2x2 / 再変形して2x2にプーリング
        seq_map = seq.transpose(1, 2).reshape(B, C, H, W)
        pooled = F.adaptive_avg_pool2d(seq_map, (2, 2))
        retokens = pooled.flatten(2).transpose(1, 2)  # (B, 4, C)

        return retokens, seq_map


# =============================================================================
# Cross-Scale Fusion / クロススケール融合
# =============================================================================

class CrossScaleFusion(nn.Module):
    """Cross-scale fusion with cross-attention between small and big grids

    小グリッドと大グリッド間のクロスアテンションによるクロススケール融合
    """

    def __init__(self, dim: int, heads: int = 6, dropout: float = 0.0, layers: int = 2):
        super().__init__()
        # Self-attention layers for each scale / 各スケールの自己アテンションレイヤー
        self.layers_s = nn.ModuleList([
            AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0)
            for _ in range(layers)
        ])
        self.layers_b = nn.ModuleList([
            AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0)
            for _ in range(layers)
        ])

        # Cross-attention layers / クロスアテンションレイヤー
        self.cross_s = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
            for _ in range(layers)
        ])
        self.cross_b = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
            for _ in range(layers)
        ])

        self.norm_s = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, tok_s: torch.Tensor, tok_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tok_s: (B, Ts, C) small grid tokens / 小グリッドトークン
            tok_b: (B, Tb, C) big grid tokens / 大グリッドトークン
        Returns:
            (B, 2 + Ts + Tb, C) fused tokens with CLS tokens / CLSトークンを含む融合トークン
        """
        B, Ts, C = tok_s.shape
        Tb = tok_b.shape[1]

        # Add CLS tokens / CLSトークンを追加
        cls_s = tok_s.new_zeros(B, 1, C)
        cls_b = tok_b.new_zeros(B, 1, C)
        tok_s = torch.cat([cls_s, tok_s], dim=1)
        tok_b = torch.cat([cls_b, tok_b], dim=1)

        # Cross-attention between scales / スケール間のクロスアテンション
        for ls, lb, cs, cb in zip(self.layers_s, self.layers_b, self.cross_s, self.cross_b):
            # Self-attention / 自己アテンション
            tok_s = ls(tok_s)
            tok_b = lb(tok_b)

            # Cross-attention: small CLS attends to big tokens / クロスアテンション：小CLSが大トークンに注目
            q_s = self.norm_s(tok_s[:, :1])
            q_b = self.norm_b(tok_b[:, :1])

            cls_s_upd, _ = cs(
                q_s,
                torch.cat([tok_b, q_b], dim=1),
                torch.cat([tok_b, q_b], dim=1),
                need_weights=False,
            )
            cls_b_upd, _ = cb(
                q_b,
                torch.cat([tok_s, q_s], dim=1),
                torch.cat([tok_s, q_s], dim=1),
                need_weights=False,
            )

            # Update CLS tokens / CLSトークンを更新
            tok_s = torch.cat([tok_s[:, :1] + cls_s_upd, tok_s[:, 1:]], dim=1)
            tok_b = torch.cat([tok_b[:, :1] + cls_b_upd, tok_b[:, 1:]], dim=1)

        # Concatenate all tokens / 全トークンを結合
        tokens = torch.cat([tok_s[:, :1], tok_b[:, :1], tok_s[:, 1:], tok_b[:, 1:]], dim=1)
        return tokens  # (B, 2 + Ts + Tb, C)


# =============================================================================
# Tile Encoder with DINO Backbone / DINOバックボーンを使用したタイルエンコーダー
# =============================================================================

class TileEncoder(nn.Module):
    """Multi-grid tile encoder using DINO backbone

    DINOバックボーンを使用したマルチグリッドタイルエンコーダー

    Divides input image into grid of tiles and processes each with DINO.
    入力画像をタイルのグリッドに分割し、各タイルをDINOで処理します。
    """

    def __init__(self, backbone: nn.Module, input_res: int):
        super().__init__()
        self.backbone = backbone
        self.input_res = input_res

    def forward(self, x: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image / 入力画像
            grid: (r, c) grid dimensions / グリッド次元
        Returns:
            (B, r*c, feat_dim) tile features / タイル特徴
        """
        B, C, H, W = x.shape
        r, c = grid

        # Compute tile boundaries / タイル境界を計算
        hs = torch.linspace(0, H, steps=r + 1, device=x.device).round().long()
        ws = torch.linspace(0, W, steps=c + 1, device=x.device).round().long()

        # Extract and resize tiles / タイルを抽出してリサイズ
        tiles = []
        for i in range(r):
            for j in range(c):
                rs, re = hs[i].item(), hs[i + 1].item()
                cs, ce = ws[j].item(), ws[j + 1].item()
                xt = x[:, :, rs:re, cs:ce]

                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(
                        xt, size=(self.input_res, self.input_res), mode="bilinear", align_corners=False
                    )
                tiles.append(xt)

        tiles = torch.stack(tiles, dim=1)  # (B, T, C, H, W)
        flat = tiles.view(-1, C, self.input_res, self.input_res)

        # Process tiles through backbone / バックボーンでタイルを処理
        feats = self.backbone(flat)  # (B*T, feat_dim)
        feats = feats.view(B, -1, feats.shape[-1])  # (B, T, feat_dim)

        return feats


# =============================================================================
# Pyramid Mixer / ピラミッドミキサー
# =============================================================================

class PyramidMixer(nn.Module):
    """Pyramid mixer with MobileViT → PVT → Mamba stages

    MobileViT → PVT → Mambaステージを持つピラミッドミキサー

    Processes tokens through three stages with increasing abstraction.
    抽象度を増しながら3つのステージでトークンを処理します。
    """

    def __init__(
        self,
        dim_in: int,
        dims: Tuple[int, int, int],
        mobilevit_heads: int = 4,
        mobilevit_depth: int = 2,
        sra_heads: int = 6,
        sra_ratio: int = 2,
        mamba_depth: int = 3,
        mamba_kernel: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3 = dims

        # Stage 1: MobileViT (local + global fusion) / ステージ1：MobileViT（局所+グローバル融合）
        self.proj1 = nn.Linear(dim_in, c1)
        self.mobilevit = MobileViTBlock(
            c1, heads=mobilevit_heads, depth=mobilevit_depth, dropout=dropout
        )

        # Stage 2: PVT + Local Mamba / ステージ2：PVT + 局所Mamba
        self.proj2 = nn.Linear(c1, c2)
        self.pvt = PVTBlock(c2, heads=sra_heads, sr_ratio=sra_ratio, dropout=dropout, mlp_ratio=3.0)
        self.mamba_local = LocalMambaBlock(c2, kernel_size=mamba_kernel, dropout=dropout)

        # Stage 3: Global Mamba / ステージ3：グローバルMamba
        self.proj3 = nn.Linear(c2, c3)
        self.mamba_global = nn.ModuleList([
            LocalMambaBlock(c3, kernel_size=mamba_kernel, dropout=dropout)
            for _ in range(mamba_depth)
        ])
        self.final_attn = AttentionBlock(c3, heads=min(8, c3 // 64 + 1), dropout=dropout, mlp_ratio=2.0)

    def _tokens_to_map(self, tokens: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Convert tokens to 2D feature map / トークンを2D特徴マップに変換"""
        B, N, C = tokens.shape
        H, W = target_hw
        need = H * W

        # Pad if needed / 必要に応じてパディング
        if N < need:
            pad = tokens.new_zeros(B, need - N, C)
            tokens = torch.cat([tokens, pad], dim=1)
        tokens = tokens[:, :need, :]

        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
        return feat_map

    @staticmethod
    def _fit_hw(n_tokens: int) -> Tuple[int, int]:
        """Find square-ish grid that fits n_tokens / n_tokensに適合する正方形に近いグリッドを見つける"""
        h = int(math.sqrt(n_tokens))
        w = h
        while h * w < n_tokens:
            w += 1
            if h * w < n_tokens:
                h += 1
        return h, w

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            tokens: (B, N, C) input tokens / 入力トークン
        Returns:
            global_feat: (B, C3) global feature vector / グローバル特徴ベクトル
            feat_maps: dict of intermediate feature maps / 中間特徴マップの辞書
        """
        B, N, C = tokens.shape
        map_hw = (3, 4)  # ~10 tokens → 3x4 map

        # Stage 1: MobileViT / ステージ1：MobileViT
        t1 = self.proj1(tokens)
        m1 = self._tokens_to_map(t1, map_hw)
        m1 = self.mobilevit(m1)
        t1_out = m1.flatten(2).transpose(1, 2)[:, :N]

        # Stage 2: PVT + Local Mamba (downsample tokens) / ステージ2：PVT + 局所Mamba（トークンをダウンサンプル）
        t2 = self.proj2(t1_out)
        new_len = max(4, N // 2)
        t2 = t2[:, :new_len] + F.adaptive_avg_pool1d(t2.transpose(1, 2), new_len).transpose(1, 2)
        hw2 = self._fit_hw(t2.size(1))

        # Pad to fit grid / グリッドに合わせてパディング
        if t2.size(1) < hw2[0] * hw2[1]:
            pad = t2.new_zeros(B, hw2[0] * hw2[1] - t2.size(1), t2.size(2))
            t2 = torch.cat([t2, pad], dim=1)

        t2 = self.pvt(t2, hw2)
        t2 = self.mamba_local(t2)

        # Stage 3: Global Mamba / ステージ3：グローバルMamba
        t3 = self.proj3(t2)
        pooled = torch.stack([t3.mean(dim=1), t3.max(dim=1).values], dim=1)  # (B, 2, C)
        t3 = pooled

        for blk in self.mamba_global:
            t3 = blk(t3)

        t3 = self.final_attn(t3)
        global_feat = t3.mean(dim=1)  # (B, C3)

        # Store intermediate features / 中間特徴を保存
        feat_maps = {
            "stage1_map": m1.detach(),
            "stage2_tokens": t2.detach(),
            "stage3_tokens": t3.detach(),
        }

        return global_feat, feat_maps


# =============================================================================
# Main V4Model / メインV4モデル
# =============================================================================

class V4Model(nn.Module):
    """V4Model - CrossPVT_T2T_MambaDINO

    Complete V4 architecture combining all components.
    全コンポーネントを組み合わせた完全なV4アーキテクチャ。

    Architecture flow / アーキテクチャフロー:
    1. Split left/right images into multi-grid tiles (4x4 small, 2x2 big)
       左右画像をマルチグリッドタイルに分割（4x4小、2x2大）
    2. Process tiles through DINO backbone
       タイルをDINOバックボーンで処理
    3. Apply T2T retokenization on small tiles
       小タイルにT2Tリトークン化を適用
    4. Cross-scale fusion between small and big tiles
       小タイルと大タイル間のクロススケール融合
    5. PyramidMixer for hierarchical feature processing
       階層的特徴処理のためのPyramidMixer
    6. Gating mechanism for left/right fusion
       左右融合のためのゲーティングメカニズム
    7. Predict 3 targets + calculate 2 remaining
       3つのターゲットを予測 + 残り2つを計算
    """

    def __init__(
        self,
        cfg: DictConfig,
        dropout: float = 0.1,
        hidden_ratio: float = 0.35,
    ):
        """
        Args:
            cfg: Configuration object / 設定オブジェクト
            dropout: Dropout rate / ドロップアウト率
            hidden_ratio: Hidden dimension ratio for prediction heads / 予測ヘッドの隠れ次元比率
        """
        super().__init__()

        # Extract config parameters / 設定パラメータを抽出
        self.dropout = dropout
        self.hidden_ratio = hidden_ratio

        # DINO backbone parameters / DINOバックボーンパラメータ
        self.dino_candidates = cfg.model.get('dino_candidates', [
            "vit_base_patch14_dinov2",
            "vit_base_patch14_reg4_dinov2",
            "vit_small_patch14_dinov2",
        ])
        self.small_grid = tuple(cfg.model.get('small_grid', [4, 4]))
        self.big_grid = tuple(cfg.model.get('big_grid', [2, 2]))

        # T2T and Cross-fusion parameters / T2Tとクロス融合パラメータ
        self.t2t_depth = cfg.model.get('t2t_depth', 2)
        self.cross_layers = cfg.model.get('cross_layers', 2)
        self.cross_heads = cfg.model.get('cross_heads', 6)

        # Pyramid parameters / ピラミッドパラメータ
        self.pyramid_dims = tuple(cfg.model.get('pyramid_dims', [384, 512, 640]))
        self.mobilevit_heads = cfg.model.get('mobilevit_heads', 4)
        self.mobilevit_depth = cfg.model.get('mobilevit_depth', 2)
        self.sra_heads = cfg.model.get('sra_heads', 8)
        self.sra_ratio = cfg.model.get('sra_ratio', 2)
        self.mamba_depth = cfg.model.get('mamba_depth', 3)
        self.mamba_kernel = cfg.model.get('mamba_kernel', 5)

        # Build DINO backbone / DINOバックボーンを構築
        self.backbone, self.feat_dim, self.backbone_name, self.input_res = self._build_dino_backbone()

        # Build components / コンポーネントを構築
        self.tile_encoder = TileEncoder(self.backbone, self.input_res)

        self.t2t = T2TRetokenizer(
            self.feat_dim, depth=self.t2t_depth, heads=self.cross_heads, dropout=dropout
        )

        self.cross = CrossScaleFusion(
            self.feat_dim, heads=self.cross_heads, dropout=dropout, layers=self.cross_layers
        )

        self.pyramid = PyramidMixer(
            dim_in=self.feat_dim,
            dims=self.pyramid_dims,
            mobilevit_heads=self.mobilevit_heads,
            mobilevit_depth=self.mobilevit_depth,
            sra_heads=self.sra_heads,
            sra_ratio=self.sra_ratio,
            mamba_depth=self.mamba_depth,
            mamba_kernel=self.mamba_kernel,
            dropout=dropout,
        )

        # Prediction heads / 予測ヘッド
        combined = self.pyramid_dims[-1] * 2  # Left + Right features / 左 + 右特徴
        self.combined_dim = combined
        hidden = max(32, int(combined * hidden_ratio))

        def head():
            return nn.Sequential(
                nn.Linear(combined, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        # Three prediction heads / 3つの予測ヘッド
        self.head_green = head()  # Dry_Green_g
        self.head_clover = head()  # Dry_Clover_g (calculated later)
        self.head_dead = head()  # Dry_Dead_g (calculated later)

        # Gating for left/right fusion / 左右融合のためのゲーティング
        self.cross_gate_left = nn.Linear(self.pyramid_dims[-1], self.pyramid_dims[-1])
        self.cross_gate_right = nn.Linear(self.pyramid_dims[-1], self.pyramid_dims[-1])

        # Softplus for non-negative predictions / 非負予測のためのSoftplus
        self.softplus = nn.Softplus(beta=1.0)

    def _build_dino_backbone(self) -> Tuple[nn.Module, int, str, int]:
        """Build DINO backbone from timm

        timmからDINOバックボーンを構築

        Returns:
            backbone: DINO model / DINOモデル
            feat_dim: Feature dimension / 特徴次元
            backbone_name: Model name / モデル名
            input_res: Input resolution / 入力解像度
        """
        last_err = None
        for name in self.dino_candidates:
            for gp in ["token", "avg", "__default__"]:
                try:
                    if gp == "__default__":
                        m = timm.create_model(name, pretrained=True, num_classes=0)
                        gp_str = "default"
                    else:
                        m = timm.create_model(name, pretrained=True, num_classes=0, global_pool=gp)
                        gp_str = gp

                    feat = m.num_features
                    input_res = self._infer_input_res(m)

                    print(
                        f"✅ Using DINO backbone: {name} | global_pool={gp_str} | "
                        f"feat_dim={feat} | input_res={input_res}"
                    )

                    # Enable gradient checkpointing if available / 利用可能な場合は勾配チェックポイントを有効化
                    if hasattr(m, "set_grad_checkpointing"):
                        m.set_grad_checkpointing(True)

                    return m, feat, name, int(input_res)

                except Exception as e:
                    last_err = e
                    continue

        raise RuntimeError(f"Failed to create DINO backbone. Last error: {last_err}")

    @staticmethod
    def _infer_input_res(m: nn.Module) -> int:
        """Infer input resolution from model / モデルから入力解像度を推測"""
        if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "img_size"):
            isz = m.patch_embed.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])

        if hasattr(m, "img_size"):
            isz = m.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])

        dc = getattr(m, "default_cfg", {}) or {}
        ins = dc.get("input_size", None)
        if ins:
            if isinstance(ins, (tuple, list)) and len(ins) >= 2:
                return int(ins[1])
            return int(ins if isinstance(ins, (int, float)) else 224)

        return 518  # Default for DINO / DINOのデフォルト

    def _half_forward(self, x_half: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process one half (left or right) of the image

        画像の片側（左または右）を処理

        Args:
            x_half: (B, 3, H, W) image half / 画像の片側
        Returns:
            feat: (B, pyramid_dims[-1]) global feature / グローバル特徴
            feat_maps: dict of intermediate features / 中間特徴の辞書
        """
        # Encode tiles at two scales / 2つのスケールでタイルをエンコード
        tiles_small = self.tile_encoder(x_half, self.small_grid)  # (B, 16, feat_dim) for 4x4
        tiles_big = self.tile_encoder(x_half, self.big_grid)  # (B, 4, feat_dim) for 2x2

        # T2T retokenization on small tiles / 小タイルにT2Tリトークン化を適用
        t2, stage1_map = self.t2t(tiles_small, self.small_grid)  # (B, 4, feat_dim)

        # Cross-scale fusion / クロススケール融合
        fused = self.cross(t2, tiles_big)  # (B, ~10, feat_dim)

        # Pyramid processing / ピラミッド処理
        feat, feat_maps = self.pyramid(fused)  # feat: (B, pyramid_dims[-1])
        feat_maps["stage1_map"] = stage1_map

        return feat, feat_maps

    def _merge_heads(
        self, f_l: torch.Tensor, f_r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge left and right features with gating and predict targets

        ゲーティングで左右特徴を融合しターゲットを予測

        Args:
            f_l: (B, C) left feature / 左特徴
            f_r: (B, C) right feature / 右特徴
        Returns:
            total: (B, 1) Dry_Total_g prediction / Dry_Total_g予測
            gdm: (B, 1) GDM_g prediction / GDM_g予測
            green: (B, 1) Dry_Green_g prediction / Dry_Green_g予測
            f_concat: (B, C*2) concatenated features / 結合された特徴
        """
        # Gating mechanism: left modulates right and vice versa
        # ゲーティングメカニズム：左が右を変調し、右が左を変調
        g_l = torch.sigmoid(self.cross_gate_left(f_r))
        g_r = torch.sigmoid(self.cross_gate_right(f_l))

        f_l = f_l * g_l
        f_r = f_r * g_r

        # Concatenate left and right / 左右を結合
        f = torch.cat([f_l, f_r], dim=1)  # (B, C*2)

        # Predict three targets (non-negative with softplus)
        # 3つのターゲットを予測（softplusで非負）
        green_pos = self.softplus(self.head_green(f))  # (B, 1)
        clover_pos = self.softplus(self.head_clover(f))  # (B, 1)
        dead_pos = self.softplus(self.head_dead(f))  # (B, 1)

        # Calculate derived targets / 派生ターゲットを計算
        # GDM = Green + Clover
        gdm = green_pos + clover_pos  # (B, 1)
        # Total = GDM + Dead = Green + Clover + Dead
        total = gdm + dead_pos  # (B, 1)

        return total, gdm, green_pos, f

    def forward(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass

        フォワードパス

        Args:
            img_left: (B, 3, H, W) left image patch / 左画像パッチ
            img_right: (B, 3, H, W) right image patch / 右画像パッチ
            labels: Optional labels (unused, for API compatibility) / オプションのラベル（API互換性のため未使用）
            masks: Optional masks (unused, for API compatibility) / オプションのマスク（API互換性のため未使用）
            do_mixup: Unused (for API compatibility) / 未使用（API互換性のため）
            do_cutmix: Unused (for API compatibility) / 未使用（API互換性のため）

        Returns:
            dict with 'logits': (B, 5) predictions
                Order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g]
                順序：[Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g]
        """
        # Process left and right halves / 左右の片側を処理
        feat_l, feats_l = self._half_forward(img_left)  # (B, pyramid_dims[-1])
        feat_r, feats_r = self._half_forward(img_right)  # (B, pyramid_dims[-1])

        # Merge and predict / 融合して予測
        total, gdm, green, f_concat = self._merge_heads(feat_l, feat_r)

        # Calculate remaining targets / 残りのターゲットを計算
        # Clover = GDM - Green (already computed in _merge_heads as clover_pos)
        # Dead = Total - GDM (already computed in _merge_heads as dead_pos)
        # But we need to reconstruct them from total, gdm, green for compatibility
        # しかし、互換性のためにtotal、gdm、greenから再構築する必要があります
        clover = torch.clamp(gdm - green, min=0.0)  # (B, 1)
        dead = torch.clamp(total - gdm, min=0.0)  # (B, 1)

        # Pack into 5 targets in train.csv order / train.csvの順序で5つのターゲットにパック
        # Order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g]
        logits = torch.cat([clover, dead, green, total, gdm], dim=1)  # (B, 5)

        return {"logits": logits}
