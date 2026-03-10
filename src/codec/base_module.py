import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple


class InceptionDWConv2d(nn.Module):
    def __init__(self, split_indexes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        
        self.dwconv_hw = nn.Conv2d(split_indexes[1], split_indexes[1], square_kernel_size, padding=square_kernel_size//2, groups=split_indexes[1])
        self.dwconv_w = nn.Conv2d(split_indexes[2], split_indexes[2], kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=split_indexes[2])
        self.dwconv_h = nn.Conv2d(split_indexes[3], split_indexes[3], kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=split_indexes[3])
        self.split_indexes = split_indexes
        
    def forward(self, x):
        id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)    


class InceptionNeXt(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.depthconv = InceptionDWConv2d((in_ch - (in_ch // 8) * 3, in_ch // 8, in_ch // 8, in_ch // 8))
        self.conv1 = nn.Conv2d(in_ch, in_ch * 2, 1)
        self.conv2 = nn.Conv2d(in_ch * 2, in_ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.depthconv(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + shortcut
        

class GatedCNNBlock(nn.Module):
    def __init__(self, in_ch, expansion_ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        hidden = int(expansion_ratio * in_ch)
        self.fc1 = nn.Conv2d(in_ch, hidden * 2, 1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, in_ch, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1, x2 = self.fc1(x).chunk(2, 1)
        x = self.fc2(self.act(x1) * self.conv(x2))
        return x + shortcut
    
    
class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        self.dwconv = InceptionDWConv2d((dim - (dim // 8) * 3, dim // 8, dim // 8, dim // 8))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Conv2d(dim, mlp_ratio * dim * 2, 1)
        self.fc2 = nn.Conv2d(mlp_ratio * dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1, x2 = self.fc1(x).chunk(2, 1)
        x = self.fc2(self.act(x1) * x2)
        return x + shortcut


class BasicBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionNeXt(in_ch),
            GatedCNNBlock(in_ch),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
    

class AnalysisTransform_Shallow(nn.Module):
    def __init__(self):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            Downsample(3, 32),
            StarBlock(32),
            Downsample(32, 64),
            StarBlock(64),
            Downsample(64, 128),
            StarBlock(128),
            Downsample(128, 192),
            StarBlock(192),
            Downsample(192, 256),
            StarBlock(256),
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x
    

class AnalysisTransform_Moderate(nn.Module):
    def __init__(self):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            Downsample(3, 64),
            StarBlock(64),
            StarBlock(64),
            Downsample(64, 128),
            StarBlock(128),
            StarBlock(128),
            Downsample(128, 192),
            StarBlock(192),
            StarBlock(192),
            Downsample(192, 256),
            StarBlock(256),
            StarBlock(256),
            Downsample(256, 320),
            StarBlock(320),
            StarBlock(320),
        )

    def forward(self, x):
        x = self.analysis_transform(x)
        return x
    

class SynthesisTransform(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            Adapter(in_ch, 1024),
            StarBlock(1024),
            StarBlock(1024),
            Upsample(1024, 768),
            StarBlock(768),
            StarBlock(768),
            Upsample(768, 512),
            StarBlock(512),
            StarBlock(512),
            Adapter(512, 576),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x[:, :320, :, :], x[:, 320:, :, :]
    

class Adapter(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, (in_ch + out_ch) // 2, 1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, (in_ch + out_ch) // 2, 5, padding=2, groups=(in_ch + out_ch) // 2),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, out_ch, 1),
        )
        self.branch2 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)
    

class HyperAnalysis(nn.Module):
    def __init__(self, M=320) -> None:
        super().__init__()
        self.reduction = nn.Sequential(
            Downsample(M, M // 2),
            StarBlock(M // 2),
            StarBlock(M // 2),
            Downsample(M // 2, M // 2),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x
    

class HyperSynthesis(nn.Module):
    def __init__(self, M=320) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            Upsample(M // 2, M // 2),
            StarBlock(M // 2),
            StarBlock(M // 2),
            Upsample(M // 2, M),
        )

    def forward(self, x):
        x = self.increase(x)
        return x
    

class SpatialContext(nn.Module):
    def __init__(self, in_ch, depth):
        super().__init__()
        self.block = nn.Sequential(
            *[BasicBlock(in_ch) for _ in range(depth)]
        )

    def forward(self, x):
        context = self.block(x)
        return context


class RateLossOutput(NamedTuple):
    rate_loss: Tensor
    quantized_total_bpp: Tensor
    quantized_latent_bpp: Tensor
    quantized_hyper_bpp: Tensor


class TargetRateModule(nn.Module):
    def __init__(self, lambda_rate):
        super().__init__()
        self.lambda_rate = lambda_rate

    def _calc_bits_per_batch(self, likelihoods: Tensor) -> Tensor:
        batch_size = likelihoods.shape[0]
        likelihoods = likelihoods.reshape(batch_size, -1)
        return likelihoods.log().sum(1) / -math.log(2)

    def forward(
        self,
        latent_likelihoods: Tensor,
        quantized_latent_likelihoods: Tensor,
        hyper_latent_likelihoods: Tensor,
        quantized_hyper_latent_likelihoods: Tensor,
        ori_h=None,
        ori_w=None,
    ):
        # calculate bits-per-pixel for both quantized and noisy quantization
        num_pixels = ori_h * ori_w

        latent_bpp = self._calc_bits_per_batch(latent_likelihoods) / num_pixels
        quantized_latent_bpp = (self._calc_bits_per_batch(quantized_latent_likelihoods) / num_pixels)
        hyper_bpp = self._calc_bits_per_batch(hyper_latent_likelihoods) / num_pixels
        quantized_hyper_bpp = (self._calc_bits_per_batch(quantized_hyper_latent_likelihoods) / num_pixels)
        total_bpp = latent_bpp + hyper_bpp
        quantized_total_bpp = quantized_latent_bpp + quantized_hyper_bpp

        return RateLossOutput(
            rate_loss=(self.lambda_rate * total_bpp).mean(),
            quantized_total_bpp=quantized_total_bpp.detach().mean(),
            quantized_latent_bpp=quantized_latent_bpp.detach().mean(),
            quantized_hyper_bpp=quantized_hyper_bpp.detach().mean(),
        )
    