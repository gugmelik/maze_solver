import hashlib
from pathlib import Path
import re

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.io_utils import ROOT_PATH, write_json_atomic


class BaseInferencer(BaseTrainer):
    """
    Evaluate one or more model checkpoints and persist generated images.

    Reuses BaseTrainer's distributed evaluation loop without constructing an
    optimizer, criterion, or learning-rate scheduler.
    """

    def __init__(
        self,
        model,
        pipe,
        metrics,
        global_config,
        accelerator,
        dataloaders,
        logger,
        writer,
        batch_transforms,
        # inferencer args
        epochs_to_infer,
        ckpt_dir,
        exp_save_dir,
        seed,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            global_config (DictConfig): run config containing inferencer config.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """

        # Intentionally NOT calling super().__init__(): inferencer is a trimmed-down
        # trainer that doesn't need optimizer, criterion, or lr_scheduler.
        # Shared attributes (model, pipe, accelerator, writer, metrics, etc.) are
        # set manually below. When adding new shared attributes to BaseTrainer,
        # update this __init__ as well.
        self.config = global_config

        self.accelerator = accelerator
        self.device = accelerator.device

        self.logger = logger
        if accelerator.is_main_process:
            logger.info(f"Using accelerate with {accelerator.mixed_precision} dtype")
        
        self.model = model
        self.pipe = pipe
        self.batch_transforms = batch_transforms
        self.writer = writer
        # BaseTrainer uses epoch_len only to derive writer steps during
        # evaluation. Checkpoint epochs are already meaningful inference steps.
        self.epoch_len = 1

        # define dataloaders
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        self.metrics = metrics
        self.evaluation_metrics = MetricTracker()
        
        self.epochs_to_infer = epochs_to_infer
        self.ckpt_dir = self._resolve_from_root(ckpt_dir)
        nested_checkpoint_dir = self.ckpt_dir / "checkpoints"
        if nested_checkpoint_dir.is_dir():
            self.ckpt_dir = nested_checkpoint_dir
        self.exp_save_dir = self._resolve_from_root(exp_save_dir)
        self.exp_save_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.gradient_accumulation_steps = 1
        self.accelerator.wait_for_everyone()

    @staticmethod
    def _resolve_from_root(path):
        path = Path(path).expanduser()
        return path if path.is_absolute() else ROOT_PATH / path


    def inference(self):
        """
        Full inference logic

        """
        for epoch in self.epochs_to_infer:
            self._last_epoch = epoch
            result = self._inference_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            if self.accelerator.is_main_process:
                for key, value in logs.items():
                    self.logger.info(f"    {key:15s}: {value}")
            self.accelerator.wait_for_everyone()
    
    
    def _inference_epoch(self, epoch):
        """
        Inference logic for an particular echeckpoint.

        Args:
            epoch (int): number of current checkpoint.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        logs = {}
        self.is_train = False

        if self.accelerator.is_main_process:
            self.writer.set_step(epoch)
            self.writer.add_scalar("general/epoch", epoch)
            
        if epoch != 0:
            ckpt_pth = Path(self.ckpt_dir) / f"checkpoint-epoch{epoch}.pth"
            self._from_pretrained(ckpt_pth)
        
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            if self.accelerator.is_main_process:
                self._save_partition_metrics(epoch, part, val_logs)
            logs.update(**{f"{part}/{name}": value for name, value in val_logs.items()})
            self.accelerator.wait_for_everyone()

        return logs

    def _handle_evaluation_artifacts(self, epoch, part, batch, *, keep):
        """Save original generated images for non-padding dataset samples."""
        if not keep:
            return

        generated_images = batch.get("generated")
        if not generated_images:
            return

        output_dir = self._get_partition_output_dir(epoch, part)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_stem = self._get_artifact_stem(batch)

        for image_index, image in enumerate(generated_images):
            # Save the pipeline output directly. The 256x256 image created by
            # BaseTrainer is a separate logging thumbnail and is never used here.
            image.save(output_dir / f"{artifact_stem}_{image_index:03d}.jpg")

    def _get_partition_output_dir(self, epoch, part):
        return self.exp_save_dir / f"checkpoint_{epoch}" / str(part)

    def _get_artifact_stem(self, batch):
        parts = []
        if batch.get("id") is not None:
            parts.append(str(batch["id"]))
        parts.append(str(batch.get("prompt", "sample")))
        return self._sanitize_artifact_stem("__".join(parts))

    @staticmethod
    def _sanitize_artifact_stem(value, max_length=180):
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
        value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
        value = re.sub(r"\s+", "_", value).strip("._")
        readable_length = max_length - len(digest) - 2
        readable_value = value[:readable_length] or "sample"
        return f"{readable_value}__{digest}"

    def _save_partition_metrics(self, epoch, part, metrics):
        metrics = {name: float(value) for name, value in metrics.items()}
        write_json_atomic(
            metrics,
            self._get_partition_output_dir(epoch, part) / "metrics.json",
        )

    def _log_batch(self, batch_idx, batch, mode="inference"):
        # Full-resolution images are persisted by
        # _handle_evaluation_artifacts on their owning process.
        return None
