import torch
import torch.nn as nn
import torch.nn.functional as F
from src.codec.base_entropy_coder import PracticalEntropyBottleneck, PracticalGaussianConditional, EntropyCoder
from src.codec.base_module import *

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def ste_round(x):
    return (torch.round(x) - x).detach() + x


class PixelCodec(nn.Module):
    def __init__(self, codec_type = "AEIC-ME", lambda_rate = 0.):
        super().__init__()

        if codec_type == "AEIC-ME":
            M = 320
            context_dim = M * 3
            context_depth = 4
            self.g_a = AnalysisTransform_Moderate()
        elif codec_type == "AEIC-SE":
            M = 256
            context_dim = M * 2
            context_depth = 3
            self.g_a = AnalysisTransform_Shallow()
        else:
            raise NotImplementedError("Codec type not supported")

        self.y_channel = M
        self.g_s = SynthesisTransform(in_ch=M)
        self.h_a = HyperAnalysis(M=M)
        self.h_s = HyperSynthesis(M=M)

        self.adapter_in = nn.ModuleList(Adapter(M, context_dim) for _ in range(4))
        self.g_c = SpatialContext(context_dim, context_depth)
        self.adapter_out = nn.ModuleList(Adapter(context_dim, M * 2) for _ in range(4))

        # Gussian Conditional
        self.entropy_bottleneck = PracticalEntropyBottleneck(M // 2)
        self.gaussian_conditional = PracticalGaussianConditional(None)
        self.masks = {}

        self.rate = TargetRateModule(lambda_rate)
        

    def forward(self):
        pass
    

    def inference(self, x, ori_h=None, ori_w=None):
        
        y_h, y_w = x.shape[2:]
        y_h, y_w = y_h // 32, y_w // 32
        z_h, z_w = math.ceil(y_h / 4), math.ceil(y_w / 4)
        pad_h, pad_w = z_h * 4 - y_h, z_w * 4 - y_w

        y = self.g_a(x)

        y_padded = F.pad(y, pad=(0, pad_w, 0, pad_h), mode='constant')
        z = self.h_a(y_padded)

        _, quantized_z_likelihoods = self.entropy_bottleneck(z, training=False)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = torch.round(z - z_offset) + z_offset

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)[:, :, 0 : y_h, 0 : y_w]
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        means_0 = means_0_supp * mask_0
        scales_0 = scales_0_supp * mask_0
        y_0 = y * mask_0
        y_hat_0 = torch.round(y_0 - means_0) + means_0

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        means_1 = means_1_supp * mask_1
        scales_1 = scales_1_supp * mask_1
        y_1 = y * mask_1
        y_hat_1 = torch.round(y_1 - means_1) + means_1

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        means_2 = means_2_supp * mask_2
        scales_2 = scales_2_supp * mask_2
        y_2 = y * mask_2
        y_hat_2 = torch.round(y_2 - means_2) + means_2

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        means_3 = means_3_supp * mask_3
        scales_3 = scales_3_supp * mask_3
        y_3 = y * mask_3
        y_hat_3 = torch.round(y_3 - means_3) + means_3

        scales_all = scales_0 + scales_1 + scales_2 + scales_3
        means_all = means_0 + means_1 + means_2 + means_3

        _, quantized_y_likelihoods = self.gaussian_conditional(y, scales_all, means_all, training=False)

        y_hat = base * (1 - mask_3) + y_hat_3
        x_hat, res1 = self.g_s(y_hat)

        RateLossOutput = self.rate(
            latent_likelihoods=quantized_y_likelihoods,
            quantized_latent_likelihoods=quantized_y_likelihoods,
            hyper_latent_likelihoods=quantized_z_likelihoods,
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
    

    def sequeeze(self, latent):
        latent_group_1, latent_group_2, latent_group_3, latent_group_4 = latent.chunk(4, 1)
        return (latent_group_1 + latent_group_2) + (latent_group_3 + latent_group_4)
    
    
    def unsequeeze_with_mask(self, latent_sequeeze, mask):
        mask_group_1, mask_group_2, mask_group_3, mask_group_4 = mask.chunk(4, 1)
        latent = torch.cat((latent_sequeeze * mask_group_1, latent_sequeeze * mask_group_2, latent_sequeeze * mask_group_3, latent_sequeeze * mask_group_4), dim=1)
        return latent
    

    def compress_group_with_mask(self, latent, scales, means, mask, symbols_list, indexes_list):
        latent_squeeze = self.sequeeze_with_mask(latent, mask)
        scales_squeeze = self.sequeeze_with_mask(scales, mask)
        means_squeeze = self.sequeeze_with_mask(means, mask)
        indexes = self.gaussian_conditional.build_indexes(scales_squeeze)
        latent_squeeze_hat = self.gaussian_conditional.quantize(latent_squeeze, "symbols", means_squeeze)
        symbols_list.extend(latent_squeeze_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        latent_hat = self.unsequeeze_with_mask(latent_squeeze_hat + means_squeeze, mask)
        return latent_hat
    

    @torch.no_grad()
    def compress_group(self, y_q_list, scales_list):
        y = [self.sequeeze(y_q) for y_q in y_q_list]
        scales = [self.sequeeze(scales) for scales in scales_list]
        y = [t.flatten() for t in y]
        scales = [t.flatten() for t in scales]
        sizes = [t.numel() for t in scales]
        scales_cat = torch.cat(scales, dim=0)
        indexes_cat = self.my_build_indexes(scales_cat)
        indexes = list(indexes_cat.split(sizes))
        y_cpu = [t.cpu().numpy() for t in y]
        idx_cpu = [t.cpu().numpy() for t in indexes]

        return (*y_cpu, *idx_cpu)
    

    def decompress_group_with_mask(self, scales, means, mask):
        scales_squeeze = self.sequeeze(scales)
        means_squeeze = self.sequeeze(means)
        indexes = self.my_build_indexes(scales_squeeze)
        latent_squeeze_q = self.entropy_coder.decode_stream(indexes.reshape(-1), self.y_cdf_group_index)
        latent_squeeze_q = latent_squeeze_q.reshape(scales_squeeze.shape).to(scales_squeeze.device).to(scales_squeeze.dtype)
        latent_hat = self.unsequeeze_with_mask(latent_squeeze_q + means_squeeze, mask)
        return latent_hat
    

    def my_build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.my_log_scale_min) / self.my_log_scale_step
        indexes = indexes.clamp_(0, SCALES_LEVELS - 1)
        indexes = torch.where(scales < 0.08, torch.zeros_like(indexes) - 1, indexes)
        return indexes.int()


    @torch.no_grad()
    def compress(self, x):

        B, C, H, W = x.shape
        y_h, y_w = H // 32, W // 32
        z_h, z_w = math.ceil(y_h / 4), math.ceil(y_w / 4)
        pad_h = z_h * 4 - y_h
        pad_w = z_w * 4 - y_w

        masks = self.get_mask_four_parts(1, self.y_channel, y_h, y_w, device=x.device)

        indexes, z_offset = self.entropy_bottleneck.get_compress_info([1, self.y_channel // 2, z_h, z_w])

        y = self.g_a(x)
        y_padded = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
        z = self.h_a(y_padded)
        z_q = torch.round(z - z_offset)
        z_hat = z_q + z_offset
        z_q = z_q.flatten().cpu().numpy()

        base = self.h_s(z_hat)[:, :, :y_h, :y_w]
        y_q_list = []
        scales_list = []

        for i in range(4):

            mask = masks[i]
            out = self.adapter_out[i](self.g_c(self.adapter_in[i](base)))
            means_supp, scales_supp = out.chunk(2, 1)
            means = means_supp * mask
            scales = scales_supp * mask
            y_part = y * mask
            y_q = torch.round(y_part - means)
            y_q_list.append(y_q)
            scales_list.append(scales)

            if i < 3:
                y_hat = y_q + means
                base = base * (1 - mask) + y_hat

        y_q_0, y_q_1, y_q_2, y_q_3, indexes_0, indexes_1, indexes_2, indexes_3 = self.compress_group(y_q_list, scales_list)

        self.entropy_coder.encode_with_z4y_indexes(
            z_q, indexes, self.z_cdf_group_index,
            y_q_0, indexes_0,
            y_q_1, indexes_1,
            y_q_2, indexes_2,
            y_q_3, indexes_3,
            self.y_cdf_group_index)
        self.entropy_coder.flush()
        
        return
    

    @torch.no_grad()
    def decompress(self, z_size, padded_y_h, padded_y_w):

        indexes, z_offset = self.entropy_bottleneck.get_compress_info(z_size)
        z_q = self.entropy_coder.decode_stream(indexes.reshape(-1), self.z_cdf_group_index)
        z_hat = z_q.reshape(indexes.shape).to(z_offset.device, dtype=z_offset.dtype) + z_offset

        masks = self.get_mask_four_parts(1, self.y_channel, padded_y_h, padded_y_w, device=z_hat.device)
        base = self.h_s(z_hat)[:, :, :padded_y_h, :padded_y_w]

        y_parts = []
        for i in range(4):

            mask = masks[i]
            out = self.adapter_out[i](self.g_c(self.adapter_in[i](base)))
            means_supp, scales_supp = out.chunk(2, 1)
            means = means_supp * mask
            scales = scales_supp * mask
            y_hat = self.decompress_group_with_mask(scales, means, mask)
            y_parts.append(y_hat)

            if i < 3:
                base = base * (1 - mask) + y_hat

        y_hat = base * (1 - masks[3]) + y_parts[3]
        x_hat, res1 = self.g_s(y_hat)

        return x_hat, res1


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.entropy_bottleneck.update(force=force)

        self.entropy_coder = EntropyCoder()

        self.y_quantized_cdf = self.gaussian_conditional.quantized_cdf.cpu().numpy()
        self.y_cdf_length = self.gaussian_conditional.cdf_length.reshape(-1).int().cpu().numpy()
        self.y_offset = self.gaussian_conditional.offset.reshape(-1).int().cpu().numpy()

        self.z_quantized_cdf, self.z_cdf_length, self.z_offset = self.entropy_bottleneck.set_cdf()

        self.my_log_scale_min = math.log(SCALES_MIN)
        self.my_log_scale_max = math.log(SCALES_MAX)
        self.my_log_scale_step = (self.my_log_scale_max - self.my_log_scale_min) / (SCALES_LEVELS - 1)

        self.z_cdf_group_index = self.entropy_coder.add_cdf(self.z_quantized_cdf, self.z_cdf_length, self.z_offset)
        self.y_cdf_group_index = self.entropy_coder.add_cdf(self.y_quantized_cdf, self.y_cdf_length, self.y_offset)

        return updated
