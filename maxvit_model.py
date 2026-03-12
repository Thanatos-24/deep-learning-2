import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=4, kernel_size=3, stride=1, 
                 se_ratio=0.25, drop_path=0., norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        self.in_dim = in_dim
        self.out_dim = out_dim
        hidden_dim = int(in_dim * expand_ratio)
        self.use_residual = stride == 1 and in_dim == out_dim

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_dim, hidden_dim, 1, bias=False)
            self.expand_norm = norm_layer(hidden_dim)
            self.expand_act = nn.SiLU()
        else:
            self.expand_conv = None

        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size, stride, 
            kernel_size // 2, groups=hidden_dim, bias=False
        )
        self.dwnorm = norm_layer(hidden_dim)
        self.dwact = nn.SiLU()

        # SE block
        se_dim = max(1, int(in_dim * se_ratio))
        self.se = SEBlock(hidden_dim, reduction=hidden_dim // se_dim)

        # Projection phase
        self.project_conv = nn.Conv2d(hidden_dim, out_dim, 1, bias=False)
        self.project_norm = norm_layer(out_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x

        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_norm(x)
            x = self.expand_act(x)

        # Depthwise convolution
        x = self.dwconv(x)
        x = self.dwnorm(x)
        x = self.dwact(x)

        # SE
        x = self.se(x)

        # Projection
        x = self.project_conv(x)
        x = self.project_norm(x)

        if self.use_residual:
            x = self.drop_path(x) + residual

        return x


class BlockAttention(nn.Module):
    def __init__(self, dim, num_heads=8, block_size=7, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        # 将特征图分割成块
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size

        # 填充到块大小的倍数
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        _, H_pad, W_pad, _ = x.shape
        num_blocks_h = H_pad // self.block_size
        num_blocks_w = W_pad // self.block_size

        # 重塑为块: (B, num_blocks_h, block_size, num_blocks_w, block_size, C)
        x = x.view(B, num_blocks_h, self.block_size, num_blocks_w, self.block_size, C)
        # 重排为: (B * num_blocks_h * num_blocks_w, block_size * block_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * num_blocks_h * num_blocks_w, self.block_size * self.block_size, C)

        # 自注意力
        qkv = self.qkv(x).reshape(B * num_blocks_h * num_blocks_w, self.block_size * self.block_size, 3, 
                                   self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * num_blocks_h * num_blocks_w, 
                                                self.block_size * self.block_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 重塑回原始形状
        x = x.view(B, num_blocks_h, num_blocks_w, self.block_size, self.block_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H_pad, W_pad, C)

        # 移除填充
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        return x


class GridAttention(nn.Module):
    def __init__(self, dim, num_heads=8, grid_size=7, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        # 计算网格大小
        grid_h = H // self.grid_size
        grid_w = W // self.grid_size

        # 填充
        pad_h = (self.grid_size - H % self.grid_size) % self.grid_size
        pad_w = (self.grid_size - W % self.grid_size) % self.grid_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        _, H_pad, W_pad, _ = x.shape
        grid_h = H_pad // self.grid_size
        grid_w = W_pad // self.grid_size

        # 重塑为网格: (B, grid_h, grid_size, grid_w, grid_size, C)
        x = x.view(B, grid_h, self.grid_size, grid_w, self.grid_size, C)
        # 转置为: (B, grid_h, grid_w, grid_size, grid_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # 重塑为: (B * grid_h * grid_w, grid_size * grid_size, C)
        x = x.view(B * grid_h * grid_w, self.grid_size * self.grid_size, C)

        # 自注意力
        qkv = self.qkv(x).reshape(B * grid_h * grid_w, self.grid_size * self.grid_size, 3,
                                   self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * grid_h * grid_w, 
                                                self.grid_size * self.grid_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 重塑回原始形状
        x = x.view(B, grid_h, grid_w, self.grid_size, self.grid_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H_pad, W_pad, C)

        # 移除填充
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        return x


class MaxViTBlock(nn.Module):

    def __init__(self, dim, num_heads=8, block_size=7, grid_size=7, 
                 expand_ratio=4, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.block_attn = BlockAttention(
            dim, num_heads=num_heads, block_size=block_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        self.grid_attn = GridAttention(
            dim, num_heads=num_heads, grid_size=grid_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        # MBConv
        self.mbconv = MBConv(
            dim, dim, expand_ratio=expand_ratio, 
            stride=1, drop_path=drop_path, norm_layer=nn.BatchNorm2d
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm3 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        # 转换为 (B, H, W, C) 进行注意力计算
        B, C, H, W = x.shape
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Block Attention
        x_norm = self.norm1(x_norm)
        x_attn = self.block_attn(x_norm)
        x_attn = x_attn.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x + self.drop_path(x_attn)

        # Grid Attention
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm2(x_norm)
        x_attn = self.grid_attn(x_norm)
        x_attn = x_attn.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x + self.drop_path(x_attn)

        # MBConv
        x = self.mbconv(x)

        # MLP
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm3(x_norm)
        x_mlp = self.mlp(x_norm)
        x_mlp = x_mlp.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x + self.drop_path(x_mlp)

        return x


class MaxViTStage(nn.Module):
    def __init__(self, dim, depth, num_heads=8, block_size=7, grid_size=7,
                 expand_ratio=4, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            MaxViTBlock(
                dim=dim, num_heads=num_heads, block_size=block_size, grid_size=grid_size,
                expand_ratio=expand_ratio, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Stem(nn.Module):
 
    def __init__(self, in_chans=3, out_chans=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans // 2, 3, 2, 1, bias=False),
            norm_layer(out_chans // 2),
            nn.SiLU(),
            nn.Conv2d(out_chans // 2, out_chans // 2, 3, 1, 1, bias=False),
            norm_layer(out_chans // 2),
            nn.SiLU(),
            nn.Conv2d(out_chans // 2, out_chans, 3, 2, 1, bias=False),
            norm_layer(out_chans),
        )

    def forward(self, x):
        return self.conv(x)


class MaxViT(nn.Module):

    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 depths=[2, 2, 5, 2], dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], block_size=7, grid_size=7,
                 expand_ratio=4, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(depths)

        # Stem
        self.stem = Stem(in_chans, dims[0], norm_layer=nn.BatchNorm2d)

        # 构建 stages 和下采样层
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(self.num_stages):
            # 下采样层（除了第一个 stage）
            if i > 0:
                downsample = nn.Sequential(
                    nn.BatchNorm2d(dims[i-1]),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[i])
                )
            else:
                downsample = nn.Identity()

            self.downsamples.append(downsample)

            stage = MaxViTStage(
                dim=dims[i], depth=depths[i], num_heads=num_heads[i],
                block_size=block_size, grid_size=grid_size,
                expand_ratio=expand_ratio, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dp_rates[cur:cur + depths[i]],
                norm_layer=norm_layer
            )
            self.stages.append(stage)
            cur += depths[i]

        # 分类头
        self.norm = norm_layer(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x):
        x = self.stem(x)

        for i, (downsample, stage) in enumerate(zip(self.downsamples, self.stages)):
            x = downsample(x)
            x = stage(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # Global average pooling
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def maxvit_tiny(img_size=224, **kwargs):
    model = MaxViT(
        img_size=img_size,
        depths=[2, 2, 5, 2],
        dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        **kwargs
    )
    return model


def maxvit_small(img_size=224, **kwargs):
    model = MaxViT(
        img_size=img_size,
        depths=[2, 2, 5, 2],
        dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        **kwargs
    )
    return model


def maxvit_base(img_size=224, **kwargs):

    model = MaxViT(
        img_size=img_size,
        depths=[2, 6, 14, 2],
        dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        **kwargs
    )
    return model


def maxvit_large(img_size=224, **kwargs):
    model = MaxViT(
        img_size=img_size,
        depths=[2, 6, 14, 2],
        dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        **kwargs
    )
    return model


if __name__ == "__main__":
    # 测试代码
    model = maxvit_tiny(num_classes=1000, img_size=224)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")

