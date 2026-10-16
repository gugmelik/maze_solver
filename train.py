import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from accelerate import Accelerator

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import (
    log_launch_script,
    set_random_seed,
    setup_saving_and_logging,
)


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_train_lora")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    project_config = OmegaConf.to_container(config)
    logger = None
    writer = None
    
    if accelerator.is_main_process:
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
        log_launch_script(config, writer, logger)

    device = accelerator.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, logger)

    # build model architecture, then print to console
    model = instantiate(config.model, accelerator=accelerator, _recursive_=False)
    model.prepare_for_training()
    model.load_train_base_model()

    model.move_frozen_modules_to_device(device)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    # build optimizer, learning rate scheduler
    trainable_params = model.get_trainable_params(config)
    optimizer = instantiate(config.optimizer, params=trainable_params)
    
    if accelerator.is_main_process:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"Param group <{group['name']}>:")
            logger.info(f"  learning rate: {group['lr']}")
            logger.info(f"  weight decay:  {group['weight_decay']}")
            logger.info(f"  betas:  {group['betas']}")
            logger.info(f"  eps:  {group['eps']}")

            # list the names or number of params
            logger.info(f"  num params:    {len(group['params'])}")

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer) 

    train_dataloader = dataloaders["train"]
    distributed_model = model.get_distributed_model()
    distributed_model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        distributed_model,
        train_dataloader,
        optimizer,
        lr_scheduler,
    )
    model.set_distributed_model(distributed_model)
    dataloaders["train"] = train_dataloader

    for part, dataloader in dataloaders.items():
        if part != "train":
            dataloaders[part] = accelerator.prepare(dataloader)

    # Every process builds a pipeline over its local prepared model. Besides
    # parallelizing validation, this keeps all processes participating in the
    # same model forwards, which is required by sharded distributed backends.
    pipeline = instantiate(
        config.pipeline,
        model=model,
        accelerator=accelerator
    )
        
    trainer = instantiate(
        config.trainer,
        model=model,
        pipe=pipeline,
        accelerator=accelerator,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        global_config=config,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        _recursive_=False
    )

    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    main()
