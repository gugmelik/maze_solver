import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from accelerate import Accelerator

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging_inference


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_inference_lora")
def main(config):
    """
    Instantiate distributed inference components and evaluate checkpoints.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)
    accelerator = Accelerator()

    project_config = OmegaConf.to_container(config)
    logger = None
    writer = None

    if accelerator.is_main_process:
        logger = setup_saving_and_logging_inference(config)
        writer = instantiate(config.writer, logger, project_config)
    accelerator.wait_for_everyone()

    device = accelerator.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, logger)

    # build model architecture, then print to console
    model = instantiate(config.model, accelerator=accelerator, _recursive_=False)
    model.prepare_for_training()
    model.load_train_base_model()
    model.move_frozen_modules_to_device(device)

    metrics = []
    for metric_name in config.inference_metrics:
        metric_config = config.metrics[metric_name]
        metrics.append(instantiate(metric_config, name=metric_name, device=device))

    distributed_model = accelerator.prepare(model.get_distributed_model())
    model.set_distributed_model(distributed_model)

    for part, dataloader in dataloaders.items():
        dataloaders[part] = accelerator.prepare(dataloader)

    # Every process builds a pipeline over its local prepared model. FSDP
    # requires all ranks to participate in the same sequence of forwards.
    pipeline = instantiate(
        config.pipeline,
        model=model,
        accelerator=accelerator,
    )

    inferencer = instantiate(
        config.inferencer,
        model=model,
        pipe=pipeline,
        accelerator=accelerator,
        metrics=metrics,
        global_config=config,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        _recursive_=False,
    )

    inferencer.inference()
    accelerator.end_training()


if __name__ == "__main__":
    main()
