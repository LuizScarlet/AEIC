import os
import gc
import math
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from diffusers.utils.import_utils import is_xformers_available

from my_utils.testing_utils import parse_args_testing
from color_fix import adain_color_fix_quant
from AEIC_practical import AEIC
from my_utils.compress_utils import *


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor


def compress_one_image(net, stream_path, img_name, x, ori_h, ori_w):
    net.codec.entropy_coder.reset()
    enc_time = net.compress(x)
    strings = net.codec.entropy_coder.get_encoded_stream()
    if not os.path.exists(stream_path): os.makedirs(stream_path)
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        my_write_body(f, [ori_h, ori_w], strings)
    size = filesize(output)
    bpp = float(size) * 8 / (ori_h * ori_w)
    return bpp, enc_time


def decompress_one_image(net, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        strings, shape = my_read_body(f)
    ori_h, ori_w = shape
    padded_y_h = math.ceil(ori_h / 64) * 2
    padded_y_w = math.ceil(ori_w / 64) * 2
    padded_z_h = math.ceil(padded_y_h / 4)
    padded_z_w = math.ceil(padded_y_w / 4)
    z_size = (1, net.codec.y_channel // 2, padded_z_h, padded_z_w)
    net.codec.entropy_coder.reset()
    net.codec.entropy_coder.set_stream(strings)
    out_img, dec_time = net.decompress(z_size, padded_y_h, padded_y_w)
    out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
    out_img = (out_img * 0.5 + 0.5).float().cpu().detach()
    return out_img, dec_time


def main(args):

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    net = AEIC(sd_path=sd_path, args=args)
    net.cuda()
    net.codec.update(force=True)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
      

    images = glob.glob(args.img_path + '/*.png')
    print(f'\nFind {str(len(images))} images in {args.img_path}\n')

    R, ENC, DEC = [], [], []

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    np.random.seed(seed=0)
    torch.set_num_threads(1)
    
    for id, img_path in enumerate(images):

        print('[Processing]', img_path)
        (path, name) = os.path.split(img_path)
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.rec_path, fname+'.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 64)) * 64 - ori_h
        pad_w = (math.ceil(ori_w / 64)) * 64 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            try:
                if args.use_practical_entropy_coding:
                    bpp, enc_time = compress_one_image(net, args.bin_path, fname, img_padded, ori_h, ori_w)
                    out_img, dec_time = decompress_one_image(net, args.bin_path, fname)
                else:
                    x_hat, RateLossOutput = net.inference(img_padded, ori_h=ori_h, ori_w=ori_w)
                    x_hat = x_hat[:, :, :ori_h, :ori_w]
                    out_img = (x_hat * 0.5 + 0.5).float().cpu().detach()
                    bpp = RateLossOutput.quantized_total_bpp.item()
                    enc_time, dec_time = 0, 0
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(str(name))
                    print("CUDA out of memory. Continuing to next image.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))

        print('[BPP]', bpp)
        R.append(bpp)
        if id >=4:
            ENC.append(enc_time)
            DEC.append(dec_time)

        if args.color_fix and ori_h * ori_w > 768 ** 2:
            img = (img * 0.5 + 0.5).float().cpu().detach()
            img_pil = transforms.ToPILImage()(img[0].clamp(0.0, 1.0))
            output_pil = adain_color_fix_quant(output_pil, img_pil, 16)

        output_pil.save(outf)
        torch.cuda.empty_cache()

    print("\n========= AVG Results =========\n")
    print(f"BPP: {np.mean(R)}")
    print(f"Encoding Time: {np.mean(ENC)}")
    print(f"Decoding Time: {np.mean(DEC)}")

    gc.collect()
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    args = parse_args_testing()
    if not os.path.exists(args.rec_path): os.makedirs(args.rec_path)
    if not os.path.exists(args.bin_path): os.makedirs(args.bin_path)
    if args.codec_type == 'AEIC-ME':
        args.latent_tiled_size = 96
        args.latent_tiled_overlap = 32
    elif args.codec_type == 'AEIC-SE':
        args.latent_tiled_size = 192
        args.latent_tiled_overlap = 64
    main(args)
