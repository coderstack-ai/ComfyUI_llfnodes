"""UniFormer backbone (Sense-X, Apache-2.0 / MIT) stripped of mmcv/timm for inference."""
from __future__ import annotations

from functools import partial

import torch.nn as nn


def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class CBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act_layer=nn.GELU):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = CMlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        b, n, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.transpose(1, 2).reshape(b, n, h, w)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = _to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class UniFormerBackbone(nn.Module):
    """UniFormer-S dense backbone (global attention in stage 3/4)."""

    def __init__(
        self,
        layers=(3, 4, 8, 3),
        embed_dim=(64, 128, 320, 512),
        head_dim=64,
        mlp_ratio=4.0,
        qkv_bias=True,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = list(embed_dim)
        self.patch_embed1 = PatchEmbed(patch_size=4, in_chans=3, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])
        self.pos_drop = nn.Dropout(p=0.0)
        num_heads = [d // head_dim for d in embed_dim]
        self.blocks1 = nn.ModuleList([CBlock(embed_dim[0], mlp_ratio=mlp_ratio) for _ in range(layers[0])])
        self.norm1 = norm_layer(embed_dim[0])
        self.blocks2 = nn.ModuleList([CBlock(embed_dim[1], mlp_ratio=mlp_ratio) for _ in range(layers[1])])
        self.norm2 = norm_layer(embed_dim[1])
        self.blocks3 = nn.ModuleList(
            [
                SABlock(embed_dim[2], num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
                for _ in range(layers[2])
            ]
        )
        self.norm3 = norm_layer(embed_dim[2])
        self.blocks4 = nn.ModuleList(
            [
                SABlock(embed_dim[3], num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
                for _ in range(layers[3])
            ]
        )
        self.norm4 = norm_layer(embed_dim[3])

    def forward(self, x):
        outs = []
        x = self.pos_drop(self.patch_embed1(x))
        for blk in self.blocks1:
            x = blk(x)
        outs.append(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        outs.append(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        outs.append(self.norm3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        outs.append(self.norm4(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        return outs
