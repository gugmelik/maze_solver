import torch

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {k: [] for k in dataset_items[0]}
    
    for item in dataset_items:
        for k, v in item.items():
           result_batch[k].append(v)
            
    assert "pixel_values" in result_batch, result_batch.keys()
    result_batch["pixel_values"] = torch.stack(result_batch["pixel_values"]).to(memory_format=torch.contiguous_format).float()
    
    return result_batch

def collate_fn_val(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    return dataset_items[0]