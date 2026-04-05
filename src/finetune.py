import os
import gc
import lpips
import numpy as np
import transformers
from tqdm.auto import tqdm
from accelerate import Accelerator
from compressai.datasets import ImageFolder

import torch
import torch.utils.checkpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

import diffusers
from diffusers.utils.import_utils import is_xformers_available

from AEIC_trainer import AEIC_ME_Trainer
from my_utils.training_utils import *
from oc_losses.oc_ea_dists_loss import OCEADISTSLoss
import vision_aided_loss

        
def main(args):

    cfg = load_config(args.config)
    stage_manager = StageManager(cfg["stages"])

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path
    
    accelerator = Accelerator(gradient_accumulation_steps=cfg["gradient_accumulation_steps"])
    assert cfg["train_batch_size_total"] == accelerator.num_processes * cfg["train_batch_size_per_gpu"] * cfg["gradient_accumulation_steps"]
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    train_dataset = H5Dataset(
        args.train_dataset_2K,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(cfg["train_patch_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    test_dataset = ImageFolder(args.test_dataset, split="Kodak",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["train_batch_size_per_gpu"],
        num_workers=cfg["dataloader_num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg["dataloader_num_workers"],
        shuffle=False,
        pin_memory=True,
    )

    net = AEIC_ME_Trainer(sd_path=sd_path, args=args)

    if accelerator.is_main_process:
        main_device = accelerator.device
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    if is_xformers_available():
        net.unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_disc = vision_aided_loss.Discriminator(cv_type='dinov3', output_type='conv_multi_level', loss_type="multilevel_sigmoid_s", device="cuda")
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_dists = OCEADISTSLoss(device=accelerator.device)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    alex_lpips.requires_grad_(False)
    net_clip = CLIPLoss().cuda()

    layers_to_opt = []
    for p in net.parameters():
        if p.requires_grad:
            layers_to_opt.append(p)
    optimizer = torch.optim.AdamW(layers_to_opt, lr=5e-5)
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=2e-5)
    
    net, net_disc, optimizer, optimizer_disc, train_dataloader = accelerator.prepare(net, net_disc, optimizer, optimizer_disc, train_dataloader)
    net_dists, alex_lpips, net_clip = accelerator.prepare(net_dists, alex_lpips, net_clip)

    weight_dtype = torch.float32
    net.to(accelerator.device, dtype=weight_dtype)
    net_dists.to(accelerator.device, dtype=weight_dtype)
    alex_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    progress_bar = tqdm(range(0, cfg["max_train_steps"]), initial=0, desc="Steps", disable=not accelerator.is_local_main_process,)
    mse_loss = torch.nn.MSELoss()

    global_step = 0
    current_hp = {}
    for k, v in cfg["base"].items():
        current_hp[k] = float(v)
    full_exp_name = cfg["exp_name"] + str(int(float(current_hp["lambda_rate"])))
    while global_step <= cfg["max_train_steps"]:
        for batch in train_dataloader:

            stage, change_stage = stage_manager.get_stage(global_step)
            if change_stage:
                for k, v in stage.items():
                    if k in ["name", "end_step"]:
                        current_hp[k] = v
                    else:
                        current_hp[k] = float(v)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_hp["lr"]
                if accelerator.is_main_process:
                    print(f"[Stage Switch] Step {global_step} -> {current_hp}")
            l_acc = [net, net_disc]

            with accelerator.accumulate(*l_acc):
                x_hat, RateLossOutput = net(batch, ori_h=cfg["train_patch_size"], ori_w=cfg["train_patch_size"])
                x = batch.detach().float()
                x_hat = x_hat.float()

                loss_l2 = mse_loss(x_hat, x)
                loss_dists = net_dists(x_hat, x).mean()
                loss_clip = net_clip(x_hat, x)
                loss_adv = net_disc(x_hat, for_G=True).mean()

                loss = RateLossOutput.rate_loss * current_hp["lambda_rate"] \
                     + loss_l2 * current_hp["lambda_l2"] \
                     + loss_dists * current_hp["lambda_dists"] \
                     + loss_clip * current_hp["lambda_clip"] \
                     + loss_adv * current_hp["lambda_adv"]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

                loss_real = net_disc(x.detach(), for_real=True).mean() * current_hp["lambda_adv"]
                accelerator.backward(loss_real)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), cfg["max_grad_norm"])
                optimizer_disc.step()
                optimizer_disc.zero_grad()
                
                loss_fake = net_disc(x_hat.detach(), for_real=False).mean() * current_hp["lambda_adv"]
                accelerator.backward(loss_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), cfg["max_grad_norm"])
                optimizer_disc.step()
                optimizer_disc.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg["checkpointing_steps"] == 0 and global_step > 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"{full_exp_name}_{global_step}.pkl")
                        accelerator.unwrap_model(net).save_model(outf)
                        print(f"save {global_step}")

                    if global_step % cfg["eval_freq"] == 0:
                        progress_bar.update(cfg["eval_freq"])
                        l_rate, l_y, l_z, l_psnr, l_lpips = [], [], [], [], []
                        val_count = 0

                        for id, batch_val in enumerate(test_dataloader):
                            batch_val = batch_val.to(main_device)
                            B, C, H, W = batch_val.shape
                            assert B == 1, "Use batch size 1 for eval."

                            with torch.no_grad():
                                x_hat, RateLossOutput = accelerator.unwrap_model(net)(batch_val, ori_h=H, ori_w=W)

                                x = (batch_val * 0.5 + 0.5).float()
                                x_hat = (x_hat * 0.5 + 0.5).float()

                                loss_l2 = mse_loss(x_hat, x)
                                loss_psnr = 10 * (-torch.log(loss_l2) / np.log(10))
                                loss_lpips = alex_lpips(x_hat, x)

                                loss_R = RateLossOutput.quantized_total_bpp.detach()
                                loss_yrate = RateLossOutput.quantized_latent_bpp.detach()
                                loss_zrate = RateLossOutput.quantized_hyper_bpp.detach()

                                l_rate.append(loss_R.item())
                                l_y.append(loss_yrate.item())
                                l_z.append(loss_zrate.item())
                                l_lpips.append(loss_lpips.item())
                                l_psnr.append(loss_psnr.item())

                            if val_count < cfg["save_num"]:
                                x = x.cpu().detach()
                                x_hat = x_hat.cpu().detach()
                                combined = torch.cat([x, x_hat], dim=3)
                                output_pil = transforms.ToPILImage()(combined[0].clamp(0.0, 1.0))
                                outf = os.path.join(args.output_dir, "eval", f"val_{id}.png")
                                output_pil.save(outf)
                                val_count += 1

                        logs = {}
                        assert len(l_psnr) == 24
                        logs["R"] = np.mean(l_rate)
                        logs["Y"] = np.mean(l_y)
                        logs["Z"] = np.mean(l_z)
                        logs["PSNR"] = np.mean(l_psnr)
                        logs["LPIPS"] = np.mean(l_lpips)
                        formatted_logs = {
                            "R": f"{logs['R']:.4f}",
                            "Y": f"{logs['Y']:.4f}",
                            "Z": f"{logs['Z']:.4f}",
                            "PSNR": f"{logs['PSNR']:.2f}",
                            "LPIPS": f"{logs['LPIPS']:.3f}",
                        }
                        progress_bar.set_postfix(**formatted_logs)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_training()
    main(args)
