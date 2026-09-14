#!/usr/bin/env python3
"""
TimeSformer (Divided Space-Time Attention) for Surgical Phase Recognition

Bertasius et al., "Is Space-Time Attention All You Need for Video
Understanding?" (2021) の Divided Space-Time Attention をピュアPyTorchで
実装したもの。外部の timesformer / transformers パッケージには依存せず、
既存の cnn_rnn_hybrids.py と同じ入力形式 (B, T, C, H, W) または
(B, C, T, H, W) を受け取り、(B, num_classes) のロジットを返す。

各Transformerブロックは
  1. 時間方向Attention（同一空間位置のパッチをフレーム間で比較）
  2. 空間方向Attention（同一フレーム内のパッチ+clsトークンを比較）
  3. MLP
の順に適用する。時間方向Attentionの出力射影(temporal_fc)はゼロ初期化し、
学習開始直後はフレーム独立なViTと同じ挙動から始まるようにしている（原論文と同様）。
pretrained_cnn=True かつ embed_dim/num_headsがViT-B/16(768, 12)と一致する場合は、
torchvision の ImageNet事前学習済み ViT-B/16 から空間方向の重みを読み込む
（時間方向Attentionはゼロ初期化のまま）。
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import torchvision.models as tv_models

    TORCHVISION_AVAILABLE = True
except ImportError:
    logger.warning("torchvision not available. TimeSformerの事前学習重みは読み込めません。")
    TORCHVISION_AVAILABLE = False


class _Attention(nn.Module):
    """
    素朴なMulti-Head Self-Attention。入力 (N, L, D) の L 次元についてのみ
    Attentionを計算する。N側は「時間Attentionなら空間位置×バッチ」「空間Attention
    ならフレーム×バッチ」をまとめたバッチ次元として扱う。
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim})はnum_heads({num_heads})で割り切れる必要があります")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N, heads, L, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(N, L, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeSformerBlock(nn.Module):
    """Divided Space-Time Attentionを行う1ブロック（時間Attn -> 空間Attn -> MLP）。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_attn = _Attention(dim, num_heads, attn_dropout, dropout)
        self.temporal_fc = nn.Linear(dim, dim)

        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = _Attention(dim, num_heads, attn_dropout, dropout)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), dropout)

        # 時間Attentionの出力射影をゼロ初期化: 学習開始時は時間方向の項が0になり、
        # フレーム独立な空間ViTと等価な状態からスタートする（原論文の初期化方針）。
        nn.init.zeros_(self.temporal_fc.weight)
        nn.init.zeros_(self.temporal_fc.bias)

    def forward(self, x: torch.Tensor, num_frames: int, num_spatial_tokens: int) -> torch.Tensor:
        # x: (B, 1 + T*S, D) -- 先頭がclsトークン、残りが T*S 個のパッチトークン
        B, _, D = x.shape
        T, S = num_frames, num_spatial_tokens
        cls_tok, patch_tok = x[:, :1], x[:, 1:]  # (B,1,D), (B,T*S,D)

        # --- 時間方向Attention: 同一空間位置のパッチをフレーム間で比較 ---
        t = patch_tok.reshape(B, T, S, D).permute(0, 2, 1, 3).reshape(B * S, T, D)
        t = self.temporal_fc(self.temporal_attn(self.norm_temporal(t)))
        t = t.reshape(B, S, T, D).permute(0, 2, 1, 3).reshape(B, T * S, D)
        patch_tok = patch_tok + t

        # --- 空間方向Attention: 各フレーム内のパッチ + clsトークンで比較 ---
        cls_expanded = cls_tok.expand(B, T, D).reshape(B * T, 1, D)
        s_in = patch_tok.reshape(B, T, S, D).reshape(B * T, S, D)
        s_in = torch.cat([cls_expanded, s_in], dim=1)  # (B*T, 1+S, D)
        s_out = self.spatial_attn(self.norm_spatial(s_in))
        s_cls, s_patch = s_out[:, :1], s_out[:, 1:]

        patch_tok = patch_tok + s_patch.reshape(B, T, S, D).reshape(B, T * S, D)
        cls_tok = cls_tok + s_cls.reshape(B, T, D).mean(dim=1, keepdim=True)

        x = torch.cat([cls_tok, patch_tok], dim=1)

        # --- MLP ---
        x = x + self.mlp(self.norm_mlp(x))
        return x


class TimeSformer(nn.Module):
    """
    TimeSformer (Divided Space-Time Attention) による手術フェーズ認識モデル。

    Args:
        num_classes: 分類するフェーズ数
        img_size: 入力フレームの一辺のピクセル数（正方形を想定）
        patch_size: パッチ分割のサイズ
        num_frames: 入力シーケンスのフレーム数（dataset側のsequence_lengthと一致させる）
        embed_dim: Transformerの埋め込み次元
        depth: TimeSformerBlockの段数
        num_heads: Multi-Head Attentionのヘッド数
        mlp_ratio: MLP隠れ層のembed_dimに対する倍率
        dropout: MLP/Attention出力・位置埋め込みのDropout率
        attn_dropout: Attention重みに対するDropout率
        pretrained_cnn: Trueかつ embed_dim/num_headsがViT-B/16(768,12)と一致する場合、
            torchvisionのImageNet事前学習済みViT-B/16から空間方向の重みを読み込む
    """

    def __init__(
        self,
        num_classes: int = 13,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 30,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        pretrained_cnn: bool = True,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size は patch_size で割り切れる必要があります")

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        grid = img_size // patch_size
        self.num_spatial_tokens = grid * grid

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial_tokens + 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TimeSformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if pretrained_cnn:
            self._load_pretrained_spatial_weights()

        logger.info(
            f"Initialized TimeSformer (divided space-time attention) with {num_classes} classes, "
            f"embed_dim={embed_dim}, depth={depth}, heads={num_heads}, frames={num_frames}"
        )

    def _load_pretrained_spatial_weights(self) -> None:
        if not TORCHVISION_AVAILABLE:
            logger.warning("torchvisionが利用できないため、TimeSformerはランダム初期化で学習します。")
            return
        if self.embed_dim != 768 or self.blocks[0].spatial_attn.num_heads != 12:
            logger.warning(
                "TimeSformerのembed_dim/num_headsがViT-B/16(768,12)と異なるため、"
                "事前学習済み重みは読み込まずランダム初期化のまま学習します。"
            )
            return

        vit = tv_models.vit_b_16(weights="DEFAULT")

        with torch.no_grad():
            self.patch_embed.weight.copy_(vit.conv_proj.weight)
            self.patch_embed.bias.copy_(vit.conv_proj.bias)
            self.cls_token.copy_(vit.class_token)
            self.pos_embed.copy_(vit.encoder.pos_embedding)

            for block, vit_block in zip(self.blocks, vit.encoder.layers):
                block.norm_spatial.weight.copy_(vit_block.ln_1.weight)
                block.norm_spatial.bias.copy_(vit_block.ln_1.bias)
                block.spatial_attn.qkv.weight.copy_(vit_block.self_attention.in_proj_weight)
                block.spatial_attn.qkv.bias.copy_(vit_block.self_attention.in_proj_bias)
                block.spatial_attn.proj.weight.copy_(vit_block.self_attention.out_proj.weight)
                block.spatial_attn.proj.bias.copy_(vit_block.self_attention.out_proj.bias)

                block.norm_mlp.weight.copy_(vit_block.ln_2.weight)
                block.norm_mlp.bias.copy_(vit_block.ln_2.bias)
                block.mlp.net[0].weight.copy_(vit_block.mlp[0].weight)
                block.mlp.net[0].bias.copy_(vit_block.mlp[0].bias)
                block.mlp.net[3].weight.copy_(vit_block.mlp[3].weight)
                block.mlp.net[3].bias.copy_(vit_block.mlp[3].bias)

            self.norm.weight.copy_(vit.encoder.ln.weight)
            self.norm.bias.copy_(vit.encoder.ln.bias)

        del vit
        logger.info("ViT-B/16の事前学習済み重みを空間方向のAttention/MLPに読み込みました（時間方向は未学習のまま）。")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) または (B, C, T, H, W)

        Returns:
            torch.Tensor: (B, num_classes) のロジット
        """
        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)  # -> (B, T, C, H, W)

        B, T, C, H, W = x.shape
        if T != self.num_frames:
            raise ValueError(
                f"TimeSformerはnum_frames={self.num_frames}で初期化されていますが、入力のフレーム数はT={T}です。"
            )

        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)  # (B*T, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B*T, S, D)
        S = x.shape[1]

        x = x + self.pos_embed[:, 1:, :]  # 空間位置埋め込み（clsトークン分を除く）
        x = x.reshape(B, T, S, self.embed_dim) + self.time_embed[:, :T, :].unsqueeze(2)
        x = x.reshape(B, T * S, self.embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x, num_frames=T, num_spatial_tokens=S)

        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)


def create_timesformer_model(
    model_name: str,
    num_classes: int = 13,
    num_frames: int = 30,
    pretrained_cnn: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create TimeSformer models.

    Args:
        model_name: "timesformer"
        num_classes: 分類クラス数
        num_frames: 入力シーケンスのフレーム数
        pretrained_cnn: ViT-B/16の事前学習済み重みを空間方向に読み込むかどうか
        **kwargs: TimeSformerへ渡すその他の引数（embed_dim, depth, num_heads等）

    Returns:
        nn.Module: TimeSformerモデル

    Raises:
        ValueError: model_nameが未対応の場合
    """
    model_name = model_name.lower()

    if model_name == "timesformer":
        return TimeSformer(
            num_classes=num_classes,
            num_frames=num_frames,
            pretrained_cnn=pretrained_cnn,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported TimeSformer model: {model_name}")


# Model registry for easy access
TIMESFORMER_MODELS = {
    "timesformer": TimeSformer,
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_timesformer_model(
        "timesformer",
        num_classes=13,
        num_frames=8,
        embed_dim=192,
        depth=2,
        num_heads=4,
        pretrained_cnn=False,  # テスト時は事前学習重みのダウンロードを避ける
    ).to(device)

    x = torch.randn(2, 8, 3, 224, 224).to(device)
    output = model(x)
    print(f"timesformer output shape: {output.shape}")
