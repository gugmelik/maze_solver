#!/usr/bin/env bash
set -euxo pipefail

ENV_PREFIX=/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/bobkov_persongen2
WORKDIR=/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/text_separation_curs_new

cd "$WORKDIR"

CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/home/jovyan/bobkov/hf_cache \
    TORCH_DISTRIBUTED_DEBUG=INFO \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HYDRA_FULL_ERROR=1 \
    "$ENV_PREFIX/bin/accelerate" launch \
    --config_file=src/configs/ddp/ddp.yaml \
    --mixed_precision=bf16 \
    --num_processes=1 \
    inference.py --config-name=flux_inference_lora \
    ddp=ddp \
    inferencer.exp_save_dir="results/test_flux_fsdp_new" \
    inferencer.ckpt_dir="saved/test_flux_fsdp_new" \
    inferencer.epochs_to_infer='[9]' \
    datasets.val.db_liv_small.data_path=/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/dreambooth/dataset/dog2 \
    datasets.val.db_liv_small.class_name=dog \
    val_datasets_names='["db_liv_small"]' \
    model.rank=16 \
    validation_args.num_images_per_prompt=3 \
    writer=console \
    writer.run_name="test_flux_fsdp_new_inference" \
    writer.project_name="test_flux2"
