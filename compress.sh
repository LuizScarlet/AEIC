#!/bin/bash

python /src/compress.py \
    --sd_path="<PATH_TO_SD_TURBO>/sd-turbo" \
    --img_path="<PATH_TO_DATASET>/Kodak" \
    --rec_path="<PATH_TO_SAVE_OUTPUTS>/rec" \
    --bin_path="<PATH_TO_SAVE_OUTPUTS>/bin" \
    --codec_type="AEIC-SE" \
    --codec_path="<PATH_TO_AEIC>/AEIC_SE_ft2.pkl" \
    --vae_decoder_path="<PATH_TO_VAE_DECODER>/halfDecoder.ckpt" \
    # --use_practical_entropy_coding
