import numpy as np
from itertools import repeat
from PIL import Image

from hydra.utils import instantiate, get_method
from src.utils.init_utils import set_worker_seed
from copy import deepcopy


IMG_EXTENTIONS = set([
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"
])


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device, logger):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    if "train_dataset_name" in config:
        assert config.train_dataset_name in config.datasets.train, f"Choosen dataset {config.train_dataset_name} not in config. Available datasets: {config.datasets.train.keys()}"
        train_dataset = instantiate(config.datasets.train[config.train_dataset_name])  # instance transforms are defined inside
    
    val_datasets = {}
    for val_dataset_name in config.val_datasets_names:
        assert val_dataset_name in config.datasets.val, f"Choosen dataset {val_dataset_name} not in config. Available datasets: {config.datasets.val.keys()}"
        val_dataset = instantiate(config.datasets.val[val_dataset_name])
        val_datasets[val_dataset_name] = val_dataset

    # dataloaders init
    dataloaders = {}

    if "train_dataset_name" in config:
        assert config.dataloaders["train"].batch_size <= len(train_dataset), (
            f"The train batch size ({config.dataloaders['train'].batch_size}) cannot "
            f"be larger than the train dataset length ({len(train_dataset)})"
        )
                
        dataloaders["train"] = instantiate(
            config.dataloaders["train"],
            dataset=train_dataset,
            collate_fn=get_method(config.collate_fn_train),
            worker_init_fn=set_worker_seed,
        )

    
    for val_dataset_name, val_dataset in val_datasets.items():
        if val_dataset_name in config.dataloaders:
            val_dataloadre_config = config.dataloaders[val_dataset_name]
        else:
            if logger:
                logger.warning(f"Can't find config for dataloader '{val_dataset_name}', using 'val_default'")
            val_dataloadre_config = config.dataloaders["val_default"]
        
        assert val_dataloadre_config.batch_size <= len(val_dataset), (
            f"The batch size ({val_dataloadre_config.batch_size}) cannot "
            f"be larger than the dataset length ({len(val_dataset)})"
        )

        val_dataloader = instantiate(
            val_dataloadre_config,
            dataset=val_dataset,
            collate_fn=get_method(config.collate_fn_val),
            drop_last=False,
            shuffle=False,
            worker_init_fn=set_worker_seed,
        )

        dataloaders[val_dataset_name] = val_dataloader

    return dataloaders, batch_transforms


def get_bigger_crop(img, face_crop, scale=0.2):
    # to square crop 
    crop = deepcopy(face_crop)
    if crop[3] - crop[1] < crop[2] - crop[0]:
        diff = crop[2] - crop[0] - (crop[3] - crop[1])
        if diff % 2 != 0:
            crop[0] -= 1
            diff += 1
        crop[3] += diff // 2
        crop[1] -= diff // 2
    elif crop[2] - crop[0] < crop[3] - crop[1]:
        diff = crop[3] - crop[1] - (crop[2] - crop[0])
        if diff % 2 != 0:
            crop[1] -= 1
            diff += 1
        crop[2] += diff // 2
        crop[0] -= diff // 2
    assert crop[3] - crop[1] == crop[2] - crop[0], crop

    # upscale crop
    to_add = int((crop[3] - crop[1]) * scale)
    h, w, _ = np.array(img).shape
    crop = [max(0, crop[0] - to_add), max(0, crop[1] - to_add), min(w, crop[2] + to_add), min(h, crop[3] + to_add)]
    cropped_arr = np.array(img)[crop[1]:crop[3], crop[0]:crop[2]]
    return Image.fromarray(cropped_arr)


def get_crop_values(img_data, target_res=1024):
    body_crop = img_data["body_crop"]
    body_crop = [body_crop[0], body_crop[2], body_crop[1], body_crop[3]]
    crop_size = body_crop[1] - body_crop[0]

    coef = target_res / crop_size

    new_body_crop = np.array(body_crop) * coef
    new_body_crop = new_body_crop.astype(np.int64)

    x1 = new_body_crop[0]
    y1 = new_body_crop[2]
    return y1, x1