import argparse
import sys
import os
from utils.utils import InputPadder
import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import *
import torch
from .spynet_arch import SpyNet

class SpyNetFlowWrapper(nn.Module):
    def __init__(self, spynet_ckpt=None, min_size=30):
        super().__init__()
        self.spynet = SpyNet(load_path=spynet_ckpt).eval()
        self.min_size = min_size  # 用于小图像上采样以保证尺寸稳定

    @torch.no_grad()
    def forward(self, frame1, frame2):
        """
        Args:
            frame1, frame2: (B, 3, H, W), 已归一化的事件图像表示，或范围 [0,1] 的图像数据
        Returns:
            flow: (B, 2, H, W) → 表示从 frame1 到 frame2 的光流
        """
        B, C, H, W = frame1.shape
        upscale = False

        if H < self.min_size or W < self.min_size:
            upscale = True
            original_size = (H, W)
            frame1 = F.interpolate(frame1, size=(self.min_size, self.min_size), mode='bilinear', align_corners=False)
            frame2 = F.interpolate(frame2, size=(self.min_size, self.min_size), mode='bilinear', align_corners=False)

        # 直接调用 SpyNet 的 forward
        flow = self.spynet(frame1, frame2)

        if upscale:
            flow = F.interpolate(flow, size=original_size, mode='bilinear', align_corners=False)
            flow[:, 0, :, :] *= original_size[1] / self.min_size
            flow[:, 1, :, :] *= original_size[0] / self.min_size

        return flow


def flow_warp(x, flow, mode='bilinear', padding_mode='zeros', align_corners=True):
    B, C, H, W = x.size()

    # Create normalized mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)  # (H, W, 2)

    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
    grid = grid + flow.permute(0, 2, 3, 1)  # Add flow to base grid

    # Normalize grid to [-1, 1]
    grid[..., 0] = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0

    # Sample from input image
    output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return output


class Multi_Branch(nn.Module):
    def __init__(self, n_c, n_b, scale, in_c=3 * 3, branch_c=3
                 ):
        super(Multi_Branch, self).__init__()
        pad = (1, 1)
        self.scale = scale
        self.optical_flow = SpyNetFlowWrapper(min_size=256)
        self.expand_x = nn.Conv2d(9, 128, kernel_size=1, bias=False)
        self.compress = nn.Conv2d(131,128,kernel_size=1, bias=False)
        self.CAF = ChannelAttentionBlock(dim_q=128,dim_kv=128,dim_out=128,num_heads=4,
                                         bias=False,LayerNorm_type='WithBias',
                                         reduction=True,ch_compress=True )

        self.aux_feature = nn.Conv2d(branch_c, n_c, 3, 1, padding=pad)
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.ResBlock1 = make_layer(basic_block, n_b)
        self.ResBlock2 = make_layer(basic_block, n_b)

        self.main_feature = nn.Conv2d(140, n_c, 3, 1, padding=pad)
        self.conv_h_m = nn.Conv2d(n_c, n_c, 3, 1, padding=pad)
        self.conv_o_m = nn.Conv2d(n_c * 2, scale ** 2 * 3, 3, 1, padding=pad)

        self.FFM_pos = Feature_fusion_module(in_channel=n_c * 2, out_channel=n_c)
        self.FFM_neg = Feature_fusion_module(in_channel=n_c * 2, out_channel=n_c)
        self.FEM = Feature_Exchange_Module(n_c)

        initialize_weights([self.main_feature, self.conv_h_m, self.conv_o_m,
                            self.aux_feature], 0.1)

    def forward(self, xs, hs, os, f1, f2):
        x_m, x_pos, x_neg = xs
        flow = self.optical_flow(f1, f2)
        hs_warped = flow_warp(hs, flow)
        x_m=self.expand_x(x_m)
        f1_expand=F.interpolate(f1, scale_factor=self.scale, mode='nearest')
        os_final = F.pixel_shuffle(os, self.scale) + F.interpolate(f2, scale_factor=self.scale, mode='nearest')
        diff=os_final-f1_expand
        max_pool = F.max_pool2d(diff, kernel_size=3, stride=1, padding=1)
        avg_pool = F.avg_pool2d(diff, kernel_size=3, stride=1, padding=1)
        attention_map = max_pool + avg_pool
        attention_map= F.interpolate(attention_map, scale_factor=1/self.scale, mode='bilinear', align_corners=False)
        information = torch.cat((hs_warped,attention_map),dim=1)
        information = self.compress(information)
        output = self.CAF(x_m,information)

        x_m = torch.cat((output, os), dim=1)
        x_m = F.relu(self.main_feature(x_m))

        x_pos = F.relu(self.aux_feature(x_pos))
        x_pos = self.ResBlock1(self.FFM_pos(x_m, x_pos))

        x_neg = F.relu(self.aux_feature(x_neg))
        x_neg = self.ResBlock1(self.FFM_neg(x_m, x_neg))

        x_pos, x_neg = self.FEM(x_pos, x_neg, 1, 1)
        x_pos = self.ResBlock2(x_pos)
        x_neg = self.ResBlock2(x_neg)

        x_h_m = F.relu(self.conv_h_m(x_m))
        x_o_m = self.conv_o_m(torch.cat((x_pos, x_neg), dim=1))

        return x_h_m, x_o_m


class RMFNet(nn.Module):
    def __init__(self, scale, n_c, n_b, repeat=1):
        super(RMFNet, self).__init__()
        self.Multi_Branch = Multi_Branch(n_c, n_b, scale, in_c=3 * 3)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.repeat = repeat
        self.conv2 = nn.Conv2d(3, 48, 1)

    def forward(self, x, x_h, x_o, init):

        _, _, T, _, _ = x.shape
        f1 = x[:, :, 0, :, :]
        f2 = x[:, :, 1, :, :]
        f3 = x[:, :, 2, :, :]

        x_input_main = torch.cat((f1, f2, f3), dim=1)
        x_input_aux1 = torch.cat(
            (f1[:, 0:1, :, :].repeat(1, self.repeat, 1, 1), f2[:, 0:1, :, :].repeat(1, self.repeat, 1, 1),
             f3[:, 0:1, :, :].repeat(1, self.repeat, 1, 1)), dim=1)  # pos
        x_input_aux2 = torch.cat(
            (f1[:, 1:2, :, :].repeat(1, self.repeat, 1, 1), f2[:, 1:2, :, :].repeat(1, self.repeat, 1, 1),
             f3[:, 1:2, :, :].repeat(1, self.repeat, 1, 1)), dim=1)  # neg

        if init:
            x_h, x_o = self.Multi_Branch([x_input_main, x_input_aux1, x_input_aux2], x_h, x_o, f1, f2)


        else:
            x_o = self.down(x_o)
            x_h, x_o = self.Multi_Branch([x_input_main, x_input_aux1, x_input_aux2], x_h, x_o, f1, f2)
        x_o = F.pixel_shuffle(x_o, self.scale) + F.interpolate(f2, scale_factor=self.scale, mode='nearest')

        return x_h, x_o
