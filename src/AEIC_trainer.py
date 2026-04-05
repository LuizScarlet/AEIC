import sys
sys.path.append("..")

import time
import torch
import torch.nn as nn

from base_model import OneStepDiffusionDecoder
from codec.codec_trainer import PixelCodec_ME_Trainer


class AEIC_ME_Trainer(OneStepDiffusionDecoder):
    def __init__(self, sd_path=None, args=None):
        super().__init__(sd_path=sd_path, args=args)
        self.add_unet_LoRA(lora_rank_unet=args.lora_rank_unet)
        self.codec = PixelCodec_ME_Trainer()
        if args.codec_path is not None:
            self.load_AEIC_state_dict(args.codec_path)
        self.set_training_mode(args.compile_model)


    def set_training_mode(self, compile_model):
        self.unet.train()
        self.vae.train()
        self.codec.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        self.unet.conv_out.requires_grad_(True)
        
        if compile_model:
            self.codec.compile()
            self.vae.decoder.compile()
            self.unet.compile()


    def load_AEIC_state_dict(self, aeic_state_dict_path):

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
                    raise KeyError(f"Expecting unet key: {k} not found in provided state_dict_unet")
                unet_random_weights[k] = pretrained_weights["state_dict_unet"][k]
        self.unet.load_state_dict(unet_random_weights)


    def forward(self, x, ori_h=None, ori_w=None):
        l_T, RateLossOutput, residual = self.codec(x, ori_h=ori_h, ori_w=ori_w)
        l_0 = self.unet(l_T) + residual
        x_hat = self.vae.decoder(l_0).clamp(-1, 1)
        return x_hat, RateLossOutput

    
    def save_model(self, outf):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k or "conv_out" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.codec.state_dict().items()}
        torch.save(sd, outf)
    