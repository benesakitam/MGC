import os
import torch

from timm.models.layers import DropPath, trunc_normal_
from torch import nn, einsum

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from semantic.custom_fun import load_checkpoint
    has_mmseg = True
except ImportError:
    print('Please Install mmsegmentation first for semantic segmentation.')
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    has_mmdet = True
except ImportError:
    print('Please Install mmdetection first for object detection and instance segmentation.')
    has_mmdet = False


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CycleFC(nn.Module):
    """
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class CycleMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)

        # MGC-B
        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)

        # MGC-L
        # self.sfc_h = CycleFC(dim, dim, (1, 11), 1, 0)
        # self.sfc_w = CycleFC(dim, dim, (11, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)  # 3, B, 1, 1, 64

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CycleBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam

        x = x.permute(0, 3, 1, 2)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0., skip_lam=1.0, mlp_fn=CycleMLP, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(CycleBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)

    return blocks


class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        drop_rate,
        skip_lam,
        dim,
        dim_out,
        proj_factor,
        downsample,
        activation=nn.ReLU()
    ):
        super().__init__()

        # shortcut
        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion
        attn_dim = dim_out // proj_factor

        self.net = nn.Sequential(
            nn.Conv2d(dim, attn_dim, 1, bias=False),
            nn.BatchNorm2d(attn_dim),
            activation,
            CycleBlock(attn_dim, mlp_ratio=4., drop_path=drop_rate, skip_lam=skip_lam, mlp_fn=CycleMLP),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attn_dim),
            activation,
            nn.Conv2d(attn_dim, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        # init last batch norm gamma to zero
        nn.init.zeros_(self.net[-1].weight)

        # final activation
        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class MlpsStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_out=2048,
        proj_factor=4,
        num_layers=3,
        downsample=True,
        drop_path_rate=0.1,
        activation=nn.ReLU()
    ):
        super().__init__()
        self.dim = dim

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample

            drp = drop_path_rate * i / (num_layers - 1)

            layers.append(BottleBlock(
                drop_rate=drp,
                skip_lam=1.0,
                dim=dim,
                dim_out=dim_out,
                proj_factor=proj_factor,
                downsample=layer_downsample,
                activation=activation
            ))

        self.net = nn.Sequential(*layers)
        print("Successful load the MLPsStack Head")

    def forward(self, x):
        _, c, _, _ = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        return self.net(x)