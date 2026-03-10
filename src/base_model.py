import sys
sys.path.append("..")

import types
import torch
import torch.nn as nn
from peft import LoraConfig
import numpy as np
from numpy import pi, exp, sqrt
from my_utils.vaehook import VAEHook

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D, UNetMidBlock2DCrossAttn
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock

from custom_forward import MyUNet2DConditionModel_SD_forward, \
    MyCrossAttnDownBlock2D_SD_forward, \
    MyDownBlock2D_SD_forward, \
    MyUNetMidBlock2DCrossAttn_SD_forward, \
    MyCrossAttnUpBlock2D_SD_forward, \
    MyUpBlock2D_SD_forward, \
    MyResnetBlock2D_SD_forward, \
    MyTransformer2DModel_SD_forward, \
    MyVAEDecoder_SD_forward, \
    merge_peft_lora_layers, \
    clean_lora_wrappers, \
    my_lora_fwd


class OneStepDiffusionDecoder(nn.Module):
    def __init__(self, sd_path=None, args=None):
        super().__init__()

        # Stable Diffusion Models
        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae

        # Prune VAE Decoder
        ckpt_halfdecoder = torch.load(args.vae_decoder_path, weights_only=False)
        decoder = Decoder(
            in_channels=4,
            out_channels=3,
            up_block_types=["UpDecoderBlock2D" for _ in range(4)],
            block_out_channels=[64, 128, 256, 256],
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            norm_type="group",
            mid_block_add_attention=True
            ).to('cuda')
        decoder_ckpt = {}
        for k, v in ckpt_halfdecoder["state_dict"].items():
            if "decoder" in k:
                new_k = k.replace("decoder.", "")
                decoder_ckpt[new_k] = v
        decoder.load_state_dict(decoder_ckpt, strict=True)
        self.vae.decoder = decoder
        self.vae.decoder.forward = types.MethodType(MyVAEDecoder_SD_forward, self.vae.decoder)
        del self.vae.encoder, self.vae.quant_conv, self.vae.post_quant_conv
        
        # Prune Unet
        def ResnetBlock2D_remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                del module.time_emb_proj
        self.unet.apply(ResnetBlock2D_remove_time_emb_proj)
        del self.unet.time_embedding, self.unet.time_proj
        def BasicTransformerBlock_remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                del module.attn2, module.norm2
        self.unet.apply(BasicTransformerBlock_remove_cross_attn)
        def set_inplace_to_true(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True
        self.unet.apply(set_inplace_to_true)
        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_forward, module)
            elif isinstance(module, DownBlock2D):
                module.forward = types.MethodType(MyDownBlock2D_SD_forward, module)
            elif isinstance(module, UNetMidBlock2DCrossAttn):
                module.forward = types.MethodType(MyUNetMidBlock2DCrossAttn_SD_forward, module)
            elif isinstance(module, UpBlock2D):
                module.forward = types.MethodType(MyUpBlock2D_SD_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_forward, module)
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(MyResnetBlock2D_SD_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(MyTransformer2DModel_SD_forward, module)
        self.unet.apply(replace_forward_methods)
        self.unet.forward = types.MethodType(MyUNet2DConditionModel_SD_forward, self.unet)

        ### Setup conv_in & conv_out
        temp_layer = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        temp_layer.weight.data = self.unet.conv_in.weight.data.repeat(1, 80, 1, 1) / 80
        temp_layer.bias.data = self.unet.conv_in.bias.data
        self.unet.conv_in = temp_layer
        temp_layer = nn.Conv2d(320, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        temp_layer.weight.data = self.unet.conv_out.weight.data.repeat(64, 1, 1, 1) / 64
        temp_layer.bias.data = self.unet.conv_out.bias.data.repeat(64,)
        self.unet.conv_out = temp_layer
        temp_layer = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        temp_layer.weight.data = self.vae.decoder.conv_in.weight.data.repeat(1, 64, 1, 1) / 64
        temp_layer.bias.data = self.vae.decoder.conv_in.bias.data
        self.vae.decoder.conv_in = temp_layer

        if args.use_tiled_unet:
            self.forbid_tiled_unet = False
            self.latent_tiled_size = args.latent_tiled_size
            self.latent_tiled_overlap = args.latent_tiled_overlap
        else:
            self.forbid_tiled_unet = True
        if args.use_tiled_vae:
            if not hasattr(self.vae.decoder, 'original_forward'):
                setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)
            decoder = self.vae.decoder
            self.vae.decoder.forward = VAEHook(decoder, args.vae_decoder_tiled_size, is_decoder=True, fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=True)


    def forward(self):
        pass

    
    def add_unet_LoRA(self, lora_name = ["default"], lora_rank_unet = 32):

        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
        unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
        for adapter_name in lora_name:
            self.unet.add_adapter(unet_lora_config, adapter_name=adapter_name)
        self.unet_lora_layers = []
        for name, module in self.unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)


    def merge_LoRA(self, save_merged_unet_path=None):
        merge_peft_lora_layers(self.unet)
        clean_lora_wrappers(self.unet)
        if save_merged_unet_path is not None:
            torch.save(self.unet.state_dict(), save_merged_unet_path)


    @torch.inference_mode()
    def inference_unet(self, x):

        _, _, h, w = x.size()
        tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
        if h * w <= tile_size * tile_size or self.forbid_tiled_unet:
            model_pred = self.unet(x)
        else:
            print(f"[Tiled Latent]: the input latent is {h}x{w}, need to tiled")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(x.device)

            grid_rows = 0
            cur_x = 0
            while cur_x < x.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < x.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = x[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_pred = self.unet(input_list_t)
                        input_list = []
                    noise_preds.append(model_pred)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros((1, 256, h, w), device=x.device)
            contributors = torch.zeros((1, 256, h, w), device=x.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights

            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        return model_pred
    

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, 256, 1, 1))