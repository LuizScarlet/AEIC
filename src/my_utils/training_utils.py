import h5py
import yaml
import torch
import argparse
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPVisionModelWithProjection

def parse_args_training(input_args=None):

    parser = argparse.ArgumentParser()

    # pretrained weights
    parser.add_argument("--sd_path", required=True, help="Path to SD-Turbo")
    parser.add_argument("--codec_path", help="Path to pretrained AEIC weights")
    parser.add_argument("--vae_decoder_path", required=True, help="Path to pretrained pruned VAE decoder weights")

    # dataset
    parser.add_argument("--train_dataset_2K", required=True, help="Path to 2K training dataset (hdf5)")
    parser.add_argument("--train_dataset_LSDIR", help="Path to LSDIR training dataset (hdf5)")
    parser.add_argument("--test_dataset", required=True, help="Path to test dataset (Kodak)")

    # training details
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--compile_model", default=True)
    parser.add_argument("--use_tiled_vae", default=False)
    parser.add_argument("--use_tiled_unet", default=False)
    parser.add_argument("--allow_tf32", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class H5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.dataset = h5py.File(self.file_path, 'r')
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file)
        self.transform = transform
        self.keys = [key for key in self.dataset.keys() if self.dataset[key].shape[0] >= 512 and self.dataset[key].shape[1] >= 512]

    def __getitem__(self, index):
        key = self.keys[index]
        image = self.dataset[key][:]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.keys)
    

class CLIPLoss(torch.nn.Module):

    def __init__(self, clip_model_name = "openai/clip-vit-base-patch32"):
        super().__init__()
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()
        self.image_encoder.requires_grad_(False)

        self.transform_for_clip = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2.0),  
            transforms.Resize(224),                     
            transforms.CenterCrop(224),                 
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  
        ])

    def forward(self, rec, gt):

        rec_inputs = self.transform_for_clip(rec)
        gt_inputs = self.transform_for_clip(gt)

        rec_features = self.image_encoder(rec_inputs).image_embeds
        gt_features = self.image_encoder(gt_inputs).image_embeds

        rec_features = rec_features / rec_features.norm(p=2, dim=-1, keepdim=True)
        gt_features = gt_features / gt_features.norm(p=2, dim=-1, keepdim=True)

        loss = torch.norm(gt_features - rec_features, p=2, dim=-1).mean()
        return loss


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["training"]


class StageManager:
    def __init__(self, stages):
        self.stages = stages
        self.current_stage_idx = -1
        self.current_stage = None

    def get_stage(self, global_step):
        for i, stage in enumerate(self.stages):
            if global_step <= stage["end_step"]:
                if i != self.current_stage_idx:
                    self.current_stage_idx = i
                    self.current_stage = stage
                    return stage, True
                return self.current_stage, False
        return self.current_stage, False