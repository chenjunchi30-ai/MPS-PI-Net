
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np, functools, math, collections.abc
from itertools import repeat
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm
from timm.models.fx_features import register_notrace_function
from timm.models.layers import trunc_normal_, to_2tuple, DropPath
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (
             normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var((-1), keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-05) * self.weight


class WithBias_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (
             normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean((-1), keepdim=True)
        sigma = x.var((-1), keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-05) * self.weight + self.bias


class LayerNorm(nn.Module):

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[(-2)[:None]]
        return to_4d(self.body(to_3d(x)), h, w)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    else:
        return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu((self.conv1(x)), inplace=True)
        out = self.conv2(out)
        return identity + out


class SelfAttention(nn.Module):

    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SelfAttention, self).__init__()
        self.basic = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(in_channel)
        self.weight = nn.Conv2d(in_channel, (in_channel * 2), kernel_size=3, stride=1, padding=1)
        self.in_c = in_channel
        initialize_weights([self.basic, self.weight], 0.1)

    def forward(self, x):
        basic = F.relu((self.bn1(self.basic(x))), inplace=True)
        weight = self.weight(basic)
        w, b = weight[(None[:None], None[:self.in_c], None[:None], None[:None])], weight[(None[:None], self.in_c[:None], None[:None], None[:None])]
        return F.relu((w * basic + b), inplace=True)


class CrossAttention(nn.Module):

    def __init__(self, in_channel=128, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channel, (in_channel // ratio), kernel_size=1, bias=False)
        self.conv_key = nn.Conv2d(in_channel, (in_channel // ratio), kernel_size=1, bias=False)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        initialize_weights([self.conv_query, self.conv_key, self.conv_value], 0.1)

    def forward(self, x, y):
        bz, c, h, w = x.shape
        y_q = self.conv_query(y).view(bz, -1, h * w).permute(0, 2, 1)
        y_k = self.conv_key(y).view(bz, -1, h * w)
        mask = torch.bmm(y_q, y_k)
        mask = torch.softmax(mask, dim=(-1))
        x_v = self.conv_value(x).view(bz, c, -1)
        feat = torch.bmm(x_v, mask.permute(0, 2, 1))
        feat = feat.view(bz, c, h, w)
        return feat


class Feature_Exchange_Module(nn.Module):

    def __init__(self, in_channel, CA=True, ratio=8):
        super(Feature_Exchange_Module, self).__init__()
        self.CA = CA
        self.dramit1 = DRAMiTransformer(in_channel, num_head=4, chsa_head_ratio=0.25)
        self.dramit2 = DRAMiTransformer(in_channel, num_head=4, chsa_head_ratio=0.25)
        if self.CA:
            self.att1 = CrossAttention(in_channel, ratio=ratio)
            self.att2 = CrossAttention(in_channel, ratio=ratio)

    def forward(self, pos, neg, beta, gamma):
        pos = self.dramit1(pos)
        neg = self.dramit2(neg)
        feat_1 = self.att1(pos, neg)
        feat_2 = self.att2(neg, pos)
        pos_out = pos + beta * feat_1
        neg_out = neg + gamma * feat_2
        return (
         pos_out, neg_out)


class CrossChannelAttention(nn.Module):

    def __init__(self, dim_q, dim_k, num_heads, bias):
        super(CrossChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim_k, (dim_k * 2), kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d((dim_k * 2), (dim_k * 2), kernel_size=3, stride=1, padding=1, groups=(dim_k * 2), bias=bias)
        self.q = nn.Conv2d(dim_q, dim_q, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim_q, dim_q, kernel_size=1, bias=bias)
        initialize_weights([self.kv, self.kv_dwconv, self.q, self.project_out], 0.1)

    def forward(self, x, y, return_attn=False):
        b, c_kv, h, w = x.shape
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q(x)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=(self.num_heads))
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=(self.num_heads))
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=(self.num_heads))
        q = torch.nn.functional.normalize(q, dim=(-1))
        k = torch.nn.functional.normalize(k, dim=(-1))
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=(-1))
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=(self.num_heads), h=h, w=w)
        out = self.project_out(out)
        if return_attn:
            return (
             out, attn)
        return out


class ChannelAttentionBlock(nn.Module):

    def __init__(self, dim_q, dim_kv, dim_out, num_heads, bias, LayerNorm_type=None, reduction=True, ch_compress=True, squeeze_factor=4):
        super(ChannelAttentionBlock, self).__init__()
        self.ch_compress = ch_compress and squeeze_factor > 1
        if self.ch_compress:
            if LayerNorm_type is not None:
                self.norm_q = LayerNorm(dim_q // squeeze_factor, LayerNorm_type)
                self.norm_kv = LayerNorm(dim_kv // squeeze_factor, LayerNorm_type)
            self.attn = CrossChannelAttention(dim_q // squeeze_factor, dim_kv // squeeze_factor, num_heads, bias)
        else:
            if LayerNorm_type is not None:
                self.norm_q = LayerNorm(dim_q, LayerNorm_type)
                self.norm_kv = LayerNorm(dim_kv, LayerNorm_type)
            self.attn = CrossChannelAttention(dim_q, dim_kv, num_heads, bias)
        self.reduction = reduction
        if reduction:
            self.norm_out = LayerNorm(dim_kv, LayerNorm_type)
            self.ffn = nn.Sequential(nn.Conv2d((dim_kv + dim_q), dim_out, kernel_size=1, bias=bias), nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, groups=dim_out, bias=bias), nn.GELU(), nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias))
        if self.ch_compress:
            self.compress_q = nn.Conv2d(dim_q, (dim_q // squeeze_factor), kernel_size=3, stride=1, padding=1, bias=bias)
            self.compress_kv = nn.Conv2d(dim_kv, (dim_kv // squeeze_factor), kernel_size=3, stride=1, padding=1, bias=bias)
            self.expand = nn.Conv2d((dim_q // squeeze_factor), dim_q, kernel_size=3, stride=1, padding=1, bias=bias)
        initialize_weights(self.ffn, 0.1)

    def forward(self, x, y, return_attn=False):
        if self.ch_compress:
            x_compressed = self.compress_q(x)
            y_compressed = self.compress_kv(y)
            if hasattr(self, "norm_kv"):
                y_compressed = self.norm_kv(y_compressed)
            if hasattr(self, "norm_q"):
                x_compressed = self.norm_q(x_compressed)
            v = self.expand(self.attn(x_compressed, y_compressed, return_attn))
        else:
            if hasattr(self, "norm_kv"):
                y = self.norm_kv(y)
            if hasattr(self, "norm_q"):
                x = self.norm_q(x)
            v = self.attn(x, y, return_attn)
        if self.reduction:
            out = self.ffn(torch.cat([x, self.norm_out(v)], dim=1))
        return out


class Feature_fusion_module(nn.Module):

    def __init__(self, in_channel, out_channel, ratio=4):
        super(Feature_fusion_module, self).__init__()
        self.basic_block = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.local_att = nn.Sequential(nn.Conv2d(out_channel, (out_channel // ratio), 1, 1, bias=False), nn.BatchNorm2d(out_channel // ratio), nn.ReLU(inplace=True), nn.Conv2d((out_channel // ratio), out_channel, 1, 1, bias=False), nn.BatchNorm2d(out_channel))
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channel, (out_channel // ratio), 1, 1, bias=False), nn.BatchNorm2d(out_channel // ratio), nn.ReLU(inplace=True), nn.Conv2d((out_channel // ratio), out_channel, 1, 1, bias=False), nn.BatchNorm2d(out_channel))
        self.sigmoid = nn.Sigmoid()
        initialize_weights([self.basic_block, self.local_att, self.global_att], 0.1)

    def forward(self, aux, main):
        fusion = torch.cat([aux, main], dim=1)
        fusion = self.basic_block(fusion)
        local_att = self.local_att(fusion)
        global_att = self.global_att(fusion)
        main = main + fusion * self.sigmoid(local_att + global_att)
        return main


def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return "upscale_factor={}".format(self.upscale_factor)


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [
         net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_((m.weight), a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_((m.weight), a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                if isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)


class QKVProjection(nn.Module):

    def __init__(self, dim, num_head, qkv_bias=True):
        super(QKVProjection, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.qkv = nn.Conv2d(dim, (3 * dim), 1, bias=qkv_bias)

    def forward(self, x):
        B, C, H, W = x.size()
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b (l c) h w -> b l c h w", l=(self.num_head))
        return qkv

    def flops(self, resolutions):
        return resolutions[0] * resolutions[1] * 1 * 1 * self.dim * 3 * self.dim


def get_relative_position_index(win_h, win_w):
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[(None[:None], None[:None], None)] - coords_flatten[(None[:None], None, None[:None])]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[(None[:None], None[:None], 0)] += win_h - 1
    relative_coords[(None[:None], None[:None], 1)] += win_w - 1
    relative_coords[(None[:None], None[:None], 0)] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class SpatialSelfAttention(nn.Module):

    def __init__(self, dim, num_head, total_head, window_size=8, shift=0, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(SpatialSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.window_size = window_size
        self.window_area = window_size ** 2
        self.shift = shift
        self.helper = helper
        self.logit_scale = nn.Parameter((torch.log(10 * torch.ones((num_head, 1, 1)))), requires_grad=True)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head))
        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, qkv, ch=None):
        B, L, C, H, W = qkv.size()
        if self.shift > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift, -self.shift), dims=(-2, -1))
        else:
            q, k, v = rearrange(qkv, "b l c (h wh) (w ww) -> (b h w) l (wh ww) c", wh=(self.window_size),
              ww=(self.window_size)).chunk(3, dim=(-1))
            if ch is not None and self.helper:
                if self.shift > 0:
                    ch = torch.roll(ch, shifts=(-self.shift, -self.shift), dims=(-2,
                                                                                 -1))
                ch = rearrange(ch, "b (l c) (h wh) (w ww) -> (b h w) l (wh ww) c", l=(self.total_head - self.num_head),
                  wh=(self.window_size),
                  ww=(self.window_size))
                ch = torch.mean(ch, dim=1, keepdim=True)
                v = v * ch
        attn = F.normalize(q, dim=(-1)) @ F.normalize(k, dim=(-1)).transpose(2, -1)
        logit_scale = torch.clamp((self.logit_scale), max=(math.log(100.0))).exp()
        attn = attn * logit_scale
        attn = attn + self._get_rel_pos_bias()
        attn = self.attn_drop(F.softmax(attn, dim=(-1)))
        x = attn @ v
        x = window_unpartition(x, (H, W), self.window_size)
        x = self.proj_drop(self.proj(x))
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(-2, -1))
        return x

    def flops(self, resolutions):
        H, W = resolutions
        num_wins = H // self.window_size * W // self.window_size
        flops = self.num_head * H * W * self.dim if self.helper else 0
        flops += num_wins * self.num_head * self.window_area * self.dim * self.window_area
        flops += num_wins * self.num_head * self.window_area * self.window_area * self.dim
        flops += H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim
        return flops


@register_notrace_function
def window_unpartition(x, resolutions, window_size):
    return rearrange(x, "(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)", h=(resolutions[0] // window_size),
      w=(resolutions[1] // window_size),
      wh=window_size)


class ChannelSelfAttention(nn.Module):

    def __init__(self, dim, num_head, total_head, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(ChannelSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.helper = helper
        self.logit_scale = nn.Parameter((torch.log(10 * torch.ones((num_head, 1, 1)))), requires_grad=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, qkv, sp=None):
        B, L, C, H, W = qkv.size()
        q, k, v = rearrange(qkv, "b l c h w -> b l c (h w)").chunk(3, dim=(-2))
        if sp is not None:
            if self.helper:
                sp = torch.mean(sp, dim=1, keepdim=True)
                sp = rearrange(sp, "b (l c) h w -> b l c (h w)", l=1)
                v = v * sp
        attn = F.normalize(q, dim=(-1)) @ F.normalize(k, dim=(-1)).transpose(2, -1)
        logit_scale = torch.clamp((self.logit_scale), max=(math.log(100.0))).exp()
        attn = attn * logit_scale
        attn = F.softmax(attn, dim=(-1))
        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, "b l c (h w) -> b (l c) h w", h=H)
        x = self.proj_drop(self.proj(x))
        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = self.num_head * self.dim * H * W if self.helper else 0
        flops += self.num_head * self.dim * H * W * self.dim
        flops += self.num_head * self.dim * self.dim * H * W
        flops += H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim
        return flops


class ReshapeLayerNorm(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(ReshapeLayerNorm, self).__init__()
        self.dim = dim
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H)
        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        flops += H * W * self.dim
        return flops


class MobiVari1(nn.Module):

    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, (kernel_size // 2), groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        if self.dim == self.out_dim:
            return out + x
        return out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * self.kernel_size * self.kernel_size * self.dim + H * W * 1 * 1 * self.dim * self.out_dim
        return flops


class MobiVari2(MobiVari1):

    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim
        self.exp_conv = nn.Conv2d(dim, (self.expand_dim), 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, (kernel_size // 2), groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        if self.dim == self.out_dim:
            return out + x
        return out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim
        flops += H * W * self.kernel_size * self.kernel_size * self.expand_dim
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim
        return flops


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_ratio, act_layer=nn.GELU, bias=True, drop=0.0):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.hidden_ratio = hidden_ratio
        self.hidden = nn.Conv2d(dim, (int(dim * hidden_ratio)), 1, bias=bias)
        self.drop1 = nn.Dropout(drop)
        self.out = nn.Conv2d((int(dim * hidden_ratio)), dim, 1, bias=bias)
        self.drop2 = nn.Dropout(drop)
        self.act = act_layer()

    def forward(self, x):
        return self.drop2(self.out(self.drop1(self.act(self.hidden(x)))))

    def flops(self, resolutions):
        H, W = resolutions
        flops = 2 * H * W * 1 * 1 * self.dim * self.dim * self.hidden_ratio
        return flops


class NoLayer(nn.Identity):

    def __init__(self):
        super(NoLayer, self).__init__()

    def flops(self, resolutions):
        return 0

    def forward(self, x, **kwargs):
        return x.flatten(1, 2)


class DRAMiTransformer(nn.Module):

    def __init__(self, dim, num_head, chsa_head_ratio, window_size=1, shift=0, head_dim=None, qkv_bias=True, mv_ver=1, hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, helper=True, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(DRAMiTransformer, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.chsa_head = int(num_head * chsa_head_ratio)
        self.shift = shift
        self.helper = helper
        self.qkv_proj = QKVProjection(dim, num_head, qkv_bias=qkv_bias)
        self.sp_attn = SpatialSelfAttention(dim // num_head, num_head - self.chsa_head, num_head, window_size, shift, attn_drop, proj_drop, helper) if num_head - self.chsa_head != 0 else NoLayer()
        self.ch_attn = ChannelSelfAttention(dim // num_head, self.chsa_head, num_head, attn_drop, proj_drop, helper) if self.chsa_head != 0 else NoLayer()
        if mv_ver == 1:
            self.mobivari = MobiVari1(dim, 3, 1, act=mv_act)
        else:
            if mv_ver == 2:
                self.mobivari = MobiVari2(dim, 3, 1, act=mv_act, out_dim=None, exp_factor=exp_factor, expand_groups=expand_groups)
        self.norm1 = norm_layer(dim)
        self.ffn = FeedForward(dim, hidden_ratio, act_layer=act_layer)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, sp_=None, ch_=None):
        B, C, H, W = x.size()
        qkv = self.qkv_proj(x)
        sp = self.sp_attn((qkv[(None[:None], None[:self.num_head - self.chsa_head])]), ch=ch_)
        ch = self.ch_attn((qkv[(None[:None], (self.num_head - self.chsa_head)[:None])]), sp=sp_)
        attn0 = self.mobivari(torch.cat([sp, ch], dim=1))
        return attn0 + x

    def flops(self, resolutions):
        flops = self.qkv_proj.flops(resolutions)
        flops += self.sp_attn.flops(resolutions)
        flops += self.ch_attn.flops(resolutions)
        flops += self.mobivari.flops(resolutions)
        flops += self.norm1.flops(resolutions)
        flops += self.ffn.flops(resolutions)
        flops += self.norm2.flops(resolutions)
        params = sum([p.numel() for n, p in self.named_parameters()])
        return flops
