import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from src.codec.base_module import *


def ste_round(x):
    return (torch.round(x) - x).detach() + x


class PixelCodec_ME_Trainer(nn.Module):
    def __init__(self):
        super().__init__()

        M = 320
        context_dim = M * 3
        context_depth = 4
        self.g_a = AnalysisTransform_Moderate()
        self.g_s = SynthesisTransform(in_ch=M)
        self.h_a = HyperAnalysis(M=M)
        self.h_s = HyperSynthesis(M=M)

        self.adapter_in = nn.ModuleList(Adapter(M, context_dim) for _ in range(4))
        self.g_c = SpatialContext(context_dim, context_depth)
        self.adapter_out = nn.ModuleList(Adapter(context_dim, M * 2) for _ in range(4))

        # Gussian Conditional
        self.entropy_bottleneck = EntropyBottleneck(M // 2)
        self.gaussian_conditional = GaussianConditional(None)
        self.masks = {}

        self.rate = TargetRateModule()
        

    def forward(self, x, ori_h=None, ori_w=None):
        
        y = self.g_a(x)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        with torch.no_grad():
            _, quantized_z_likelihoods = self.entropy_bottleneck(z, training=False)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        means_0 = means_0_supp * mask_0
        scales_0 = scales_0_supp * mask_0
        y_0 = y * mask_0
        y_hat_0 = ste_round(y_0 - means_0) + means_0

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        means_1 = means_1_supp * mask_1
        scales_1 = scales_1_supp * mask_1
        y_1 = y * mask_1
        y_hat_1 = ste_round(y_1 - means_1) + means_1

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        means_2 = means_2_supp * mask_2
        scales_2 = scales_2_supp * mask_2
        y_2 = y * mask_2
        y_hat_2 = ste_round(y_2 - means_2) + means_2

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        means_3 = means_3_supp * mask_3
        scales_3 = scales_3_supp * mask_3
        y_3 = y * mask_3
        y_hat_3 = ste_round(y_3 - means_3) + means_3

        scales_all = scales_0 + scales_1 + scales_2 + scales_3
        means_all = means_0 + means_1 + means_2 + means_3

        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means_all)
        with torch.no_grad():
            _, quantized_y_likelihoods = self.gaussian_conditional(y, scales_all, means_all, training=False)

        y_hat = base * (1 - mask_3) + y_hat_3
        x_hat, res1 = self.g_s(y_hat)

        RateLossOutput = self.rate(
            latent_likelihoods=y_likelihoods,
            quantized_latent_likelihoods=quantized_y_likelihoods,
            hyper_latent_likelihoods=z_likelihoods,
            quantized_hyper_latent_likelihoods=quantized_z_likelihoods,
            ori_h=ori_h,
            ori_w=ori_w,
        )

        return x_hat, RateLossOutput, res1
    

    def get_mask_four_parts(self, batch, channel, height, width, device='cuda'):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_m0 = torch.tensor(((1., 0), (0, 0)), device=device)
            m0 = micro_m0.repeat((height + 1) // 2, (width + 1) // 2)
            m0 = m0[:height, :width]
            m0 = torch.unsqueeze(m0, 0)
            m0 = torch.unsqueeze(m0, 0)

            micro_m1 = torch.tensor(((0, 1.), (0, 0)), device=device)
            m1 = micro_m1.repeat((height + 1) // 2, (width + 1) // 2)
            m1 = m1[:height, :width]
            m1 = torch.unsqueeze(m1, 0)
            m1 = torch.unsqueeze(m1, 0)

            micro_m2 = torch.tensor(((0, 0), (1., 0)), device=device)
            m2 = micro_m2.repeat((height + 1) // 2, (width + 1) // 2)
            m2 = m2[:height, :width]
            m2 = torch.unsqueeze(m2, 0)
            m2 = torch.unsqueeze(m2, 0)

            micro_m3 = torch.tensor(((0, 0), (0, 1.)), device=device)
            m3 = micro_m3.repeat((height + 1) // 2, (width + 1) // 2)
            m3 = m3[:height, :width]
            m3 = torch.unsqueeze(m3, 0)
            m3 = torch.unsqueeze(m3, 0)

            m = torch.ones((batch, channel // 4, height, width), device=device)
            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]
    
