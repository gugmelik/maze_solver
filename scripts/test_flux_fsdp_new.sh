set -ex

ENV_PREFIX=/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/bobkov_persongen2
WORKDIR=/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/text_separation_curs_new


cd $WORKDIR

CUDA_VISIBLE_DEVICES=2,3 \
    HF_HOME=/home/jovyan/bobkov/hf_cache \
    COMET_API_KEY="6IbVmDWsE5nXRXP0aHvrGNHR2" COMET_WORKSPACE="retir" \
    TORCH_DISTRIBUTED_DEBUG=INFO \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HYDRA_FULL_ERROR=1 \
    accelerate launch \
    --config_file=src/configs/ddp/ddp.yaml \
    --mixed_precision=bf16 \
    --num_processes=2 \
    --gradient_accumulation_steps=1 \
    train.py --config-name=flux_train_lora \
    ddp=ddp \
    writer.run_name="test_flux_fsdp_new" \
    train_dataset_name=dreambooth \
    datasets.train.dreambooth.data_path=/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/dreambooth/dataset/dog2 \
    datasets.train.dreambooth.class_name=dog \
    datasets.val.db_liv_small.data_path=/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/bobkov/dreambooth/dataset/dog2 \
    datasets.val.db_liv_small.class_name=dog \
    val_datasets_names='["db_liv_small"]' \
    dataloaders.train.batch_size=1 \
    dataloaders.train.num_workers=1 \
    trainer.epoch_len=200 \
    trainer.n_epochs=10 \
    model.rank=16 \
    lr_for_lora=1e-4 \
    validation_args.num_images_per_prompt=3 \
    writer=console writer.project_name="test_flux2"