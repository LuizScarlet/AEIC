import sys
sys.path.append("..")

import time
import torch
import torch.nn as nn

from base_model import OneStepDiffusionDecoder
from codec.codec_practical import PixelCodec


class AEIC(OneStepDiffusionDecoder):
    def __init__(self, sd_path=None, args=None):
        super().__init__(sd_path=sd_path, args=args)
        self.add_unet_LoRA(lora_rank_unet=args.lora_rank_unet)
        self.codec = PixelCodec(codec_type=args.codec_type, lambda_rate=0.)
        self.load_AEIC_state_dict(args.codec_path, merge_LoRA=args.merge_LoRA)
        self.set_inference_mode()


    def set_inference_mode(self):
        self.unet.eval()
        self.vae.eval()
        self.codec.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(False)
        torch.compile(self.codec)
        torch.compile(self.vae.decoder)
        torch.compile(self.unet)


    def load_AEIC_state_dict(self, aeic_state_dict_path, merge_LoRA=False):

        pretrained_weights = torch.load(aeic_state_dict_path, map_location="cpu")

        codec_random_weights = self.codec.state_dict()
        pretrained_codec_keys = pretrained_weights["state_dict_codec"].keys()
        for k in codec_random_weights.keys():
            if k not in pretrained_codec_keys:
                raise KeyError(f"Expecting codec key: {k} not found in provided state_dict_codec")
            codec_random_weights[k] = pretrained_weights["state_dict_codec"][k]
        self.codec.load_state_dict(codec_random_weights)

        unet_random_weights = self.unet.state_dict()
        pretrained_unet_keys = pretrained_weights["state_dict_unet"].keys()
        for k in unet_random_weights.keys():
            if "lora" in k or "conv_in" in k or "conv_out" in k:
                if k not in pretrained_unet_keys:
                    if "default" in k:
                        revised_k = k.replace("default", "student_unet_lora")
                    elif "conv_in" in k:
                        revised_k = k.replace("conv_in", "student_conv_in")
                    elif "conv_out" in k:
                        revised_k = k.replace("conv_out", "student_conv_out")
                    else:
                        raise KeyError(f"Expecting unet key: {k} not found in provided state_dict_unet")
                    if revised_k not in pretrained_unet_keys:
                        raise KeyError(f"Expecting unet key: {k} or {revised_k} not found in provided state_dict_unet")
                    unet_random_weights[k] = pretrained_weights["state_dict_unet"][revised_k]
                else:
                    unet_random_weights[k] = pretrained_weights["state_dict_unet"][k]
        self.unet.load_state_dict(unet_random_weights)

        if merge_LoRA:
            self.merge_LoRA()


    def forward(self):
        pass


    @torch.inference_mode()
    def inference(self, x, ori_h=None, ori_w=None):
        '''Fast Inference without Bitstreams'''
        l_T, RateLossOutput, residual = self.codec.inference(x, ori_h=ori_h, ori_w=ori_w)
        l_0 = self.inference_unet(l_T) + residual
        x_hat = self.vae.decoder(l_0).clamp(-1, 1)
        return x_hat, RateLossOutput

    
    @torch.inference_mode()
    def compress(self, x):
        torch.cuda.synchronize()
        start_time = time.time()
        self.codec.compress(x)
        torch.cuda.synchronize()
        end_time = time.time()
        enc_time = end_time - start_time
        return enc_time


    @torch.inference_mode()
    def decompress(self, z_size, padded_y_h, padded_y_w):
        torch.cuda.synchronize()
        start_time = time.time()
        l_T, residual = self.codec.decompress(z_size, padded_y_h, padded_y_w)
        l_0 = self.inference_unet(l_T) + residual
        x_hat = self.vae.decoder(l_0).clamp(-1, 1)
        torch.cuda.synchronize()
        end_time = time.time()
        dec_time = end_time - start_time
        return x_hat, dec_time
    