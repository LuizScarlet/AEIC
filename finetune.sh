#!/bin/bash

accelerate launch --num_processes=2 --gpu_ids="0,1," src/finetune.py \
    --config="config/finetune_AEIC_ME.yaml" \
    --sd_path="<PATH_TO_SD_TURBO>/sd-turbo" \
    --vae_decoder_path="<PATH_TO_VAE_DECODER>/halfDecoder.ckpt" \
    --train_dataset_2K="<PATH_TO_DATASET>/dataset_2K.hdf5" \
    --test_dataset="<PATH_TO_DATASET>/Kodak" \
    --codec_path="<PATH_TO_CODEC>/AEIC_ME_pretrain_300000.pkl" \
    --output_dir="<PATH_TO_OUTPUT_DIR>"