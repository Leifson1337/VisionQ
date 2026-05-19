from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ..attention.registry import get_attention_backend
from ..core.context import AttentionContext
from ..core.tensor import SpatioTemporalTensor
from ..runtime.dispatcher import AttentionDispatcher


class VisionBackboneBlock(nn.Module):
    """Transformer block that routes projected attention through VisionQ backends."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        backend_override: str | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.backend_override = backend_override
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dispatcher = AttentionDispatcher()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(mlp_dropout),
        )
        self.attention_dropout = attention_dropout
        self.backends = nn.ModuleDict()

    def _get_backend(self, name: str) -> nn.Module:
        if name not in self.backends:
            cls = get_attention_backend(name)
            self.backends[name] = cls(
                self.dim,
                num_heads=self.num_heads,
                attn_drop=self.attention_dropout,
            )
            ref = next(self.parameters())
            self.backends[name].to(device=ref.device, dtype=ref.dtype)
        return self.backends[name]

    @staticmethod
    def _as_st_tensor(x: torch.Tensor | SpatioTemporalTensor) -> SpatioTemporalTensor:
        if isinstance(x, SpatioTemporalTensor):
            return x
        if x.dim() == 5:
            return SpatioTemporalTensor(x, modality="video")
        if x.dim() == 4:
            return SpatioTemporalTensor(x, modality="image")
        if x.dim() == 3:
            return SpatioTemporalTensor(x, modality="sequence")
        raise ValueError(f"unsupported input shape {tuple(x.shape)}")

    def forward(
        self,
        x: torch.Tensor | SpatioTemporalTensor,
        context: AttentionContext | None = None,
    ) -> SpatioTemporalTensor:
        st_x = self._as_st_tensor(x)
        context = context or AttentionContext.from_st_tensor(st_x)
        residual = st_x.flatten_all()
        x_norm = self.norm1(residual)
        batch, tokens, channels = x_norm.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv(x_norm).reshape(batch, tokens, 3, self.num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        backend_name = self.backend_override or self.dispatcher.select(context)
        out = self._get_backend(backend_name)(
            q,
            k,
            v,
            context,
            block_size=context.extra_args.get("block_size", 32),
        )
        if out.dim() != 4:
            raise RuntimeError(
                f"backend {backend_name} returned unsupported shape {tuple(out.shape)}"
            )
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        hidden = residual + self.proj_drop(self.proj(out))
        hidden = hidden + self.mlp(self.norm2(hidden))
        return SpatioTemporalTensor(hidden, st_x.modality, st_x.spatial_shape, st_x.temporal_dim)


class VisionBackbone(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        backend_override: str | None = None,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.blocks = nn.ModuleList(
            [
                VisionBackboneBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    mlp_dropout=mlp_dropout,
                    activation=activation,
                    norm_layer=norm_layer,
                    backend_override=backend_override,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor | SpatioTemporalTensor,
        context: AttentionContext | None = None,
    ) -> SpatioTemporalTensor:
        st_x = VisionBackboneBlock._as_st_tensor(x)
        for block in self.blocks:
            st_x = block(st_x, context)
        return st_x


class PatchEmbed(nn.Module):
    """Image patch embedding for `(B, C, H, W)` tensors."""

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: tuple[int, int] = (16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if image_size[0] % patch_size[0] or image_size[1] % patch_size[1]:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"PatchEmbed expects (B, C, H, W), got {tuple(x.shape)}")
        if (x.shape[2], x.shape[3]) != self.image_size:
            raise ValueError(
                f"expected image spatial shape {self.image_size}, got {(x.shape[2], x.shape[3])}"
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """Small ViT wrapper around `VisionBackbone` for supervised image tasks."""

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: tuple[int, int] = (16, 16),
        in_channels: int = 3,
        num_classes: int = 1000,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        backend_override: str | None = "flash",
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.backbone = VisionBackbone(
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            mlp_dropout=dropout,
            backend_override=backend_override,
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x) + self.pos_embed
        tokens = self.pos_drop(tokens)
        st_tokens = SpatioTemporalTensor(
            tokens,
            modality="image",
            spatial_shape=self.patch_embed.grid_size,
        )
        ctx = AttentionContext.from_st_tensor(st_tokens)
        features = self.backbone(st_tokens, ctx).flatten_all()
        return self.norm(features).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
