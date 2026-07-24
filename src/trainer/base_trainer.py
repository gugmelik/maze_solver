from abc import abstractmethod
from io import BytesIO
import os
from pathlib import Path
import random

import numpy as np
import torch
from accelerate.utils import DistributedType, broadcast_object_list, gather_object
from PIL import Image
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.model.distributed import materialize_state_dict
from src.utils.io_utils import ROOT_PATH, read_json, write_json_atomic


class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(
        self,
        model,
        pipe,
        accelerator,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        global_config,
        dataloaders,
        logger,
        writer,
        batch_transforms,
        # trainer args
        max_grad_norm,
        cfg_step,
        log_step,
        n_epochs,
        epoch_len,
        resume_from,
        from_pretrained,
        save_period,
        save_dir,
        seed=42,
        checkpoint_subdir="checkpoints",
        metrics_subdir="metrics",
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            global_config (DictConfig): experiment config containing training config.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = global_config
        self.device = accelerator.device

        self.logger = logger
        self.gradient_accumulation_steps = accelerator.gradient_accumulation_steps
        self.log_step = self._scale_batch_dependent_step(log_step)
        if accelerator.is_main_process:
            logger.info(f"Using accelerate with {accelerator.mixed_precision} dtype")

        self.model = model
        self.pipe = pipe
        self.accelerator = accelerator
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms
        self.writer = writer

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len
        self.epoch_len = self._scale_batch_dependent_step(self.epoch_len)

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = n_epochs

        self.save_period = save_period  # checkpoint each save_period epochs
        self.max_grad_norm = max_grad_norm
        self.cfg_step = self._scale_batch_dependent_step(cfg_step)
        

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker()
        self.evaluation_metrics = MetricTracker()

        # Keep run metadata and logs at the run root while storing large and
        # structured artifacts in dedicated subdirectories.
        self.run_dir = ROOT_PATH / save_dir / self.config.writer.run_name
        self.checkpoint_dir = self.run_dir / checkpoint_subdir
        self.metrics_dir = self.run_dir / metrics_subdir
        self.validation_metrics_path = self.metrics_dir / "validation.json"

        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.validation_metrics_history = (
                read_json(self.validation_metrics_path)
                if resume_from is not None and self.validation_metrics_path.exists()
                else {}
            )
            if resume_from is None:
                write_json_atomic(
                    self.validation_metrics_history,
                    self.validation_metrics_path,
                )
        self.accelerator.wait_for_everyone()

        if resume_from is not None:
            resume_path = Path(resume_from)
            if not resume_path.is_absolute():
                resume_path = self.checkpoint_dir / resume_path
                legacy_resume_path = self.run_dir / resume_from
                if not resume_path.exists() and legacy_resume_path.exists():
                    resume_path = legacy_resume_path
            self._resume_checkpoint(resume_path)

        if from_pretrained is not None:
            self._from_pretrained(from_pretrained)


    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch and evaluating it on non-train partitions
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_main_process:
                # save logged information into logs dict
                logs = {"epoch": epoch}
                logs.update(result)

                # print logged information to the screen
                for key, value in logs.items():
                    self.logger.info(f"    {key:15s}: {value}")

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        logs = {}
        self.is_train = True
        pid = os.getpid()
        self.train_metrics.reset()

        if self.accelerator.is_main_process:
            self.writer.set_step(self._scale_writer_step((epoch - 1) * self.epoch_len))
            self.writer.add_scalar("general/epoch", epoch)

        if epoch == self.start_epoch:
            logs.update(self._evaluate_all(epoch - 1))
            self.is_train = True

        self.model.train()
        
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc=f"train_{pid}", total=self.epoch_len)
        ):  
            batch["batch_idx"] = batch_idx
            batch = self.process_batch(
                batch,
                train_metrics=self.train_metrics,
            )
            
            # log current results
            if batch_idx % self.log_step == 0:
                if self.accelerator.is_main_process:
                    self.writer.set_step(self._scale_writer_step((epoch - 1) * self.epoch_len + batch_idx))
                    self.logger.debug(
                        "Train Epoch: {} {} Reduced Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )

                    lrs = self._get_lrs()
                    for part_name, part_lr in lrs.items():
                        self.writer.add_scalar(f"lrs/{part_name}", part_lr)

                    self._log_scalars(self.train_metrics, "train")
                    self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                # last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break


        # Run val/test
        logs.update(self._evaluate_all(epoch))

        return logs

    def _evaluate_all(self, epoch):
        """
        Evaluate all validation datasets and persist their averaged metrics.
        """
        logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(
                {
                    f"{part}/{name}": float(value)
                    for name, value in val_logs.items()
                }
            )

        if self.accelerator.is_main_process:
            iteration = self._scale_writer_step(epoch * self.epoch_len)
            self._save_validation_metrics(iteration, logs)
        return logs

    def _save_validation_metrics(self, iteration, metrics):
        """
        Save validation averages, indexed by optimizer iteration.
        """
        self.validation_metrics_history[str(iteration)] = metrics
        write_json_atomic(
            self.validation_metrics_history,
            self.validation_metrics_path,
        )

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.accelerator.wait_for_everyone()
        self.is_train = False
        self.evaluation_metrics.reset()
        torch.cuda.empty_cache()

        for metric in self.metrics:
            metric.to_cuda()

        was_training = self.model.training
        self.model.eval()

        if self.accelerator.is_main_process:
            self.writer.set_step(self._scale_writer_step(epoch * self.epoch_len), part)

        global_batch_idx = 0
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
                disable=not self.accelerator.is_main_process,
            ):
                batch["is_last_batch"] = batch_idx == len(dataloader) - 1
                batch_metrics = MetricTracker()
                batch = self.process_evaluation_batch(
                    batch,
                    eval_metrics=batch_metrics,
                )

                artifact_token = (self.accelerator.process_index, batch_idx)
                local_record = self._make_evaluation_record(batch, batch_metrics)
                local_record["artifact_token"] = artifact_token
                # Prepared evaluation loaders pad the final global batch so
                # every process performs the same number of forwards.
                # gather_for_metrics removes those duplicated tail samples.
                gathered_records = self.accelerator.gather_for_metrics(
                    [local_record],
                    use_gather_object=True,
                )

                keep_local_artifacts = any(
                    tuple(record["artifact_token"]) == artifact_token
                    for record in gathered_records
                )
                self._handle_evaluation_artifacts(
                    epoch,
                    part,
                    batch,
                    keep=keep_local_artifacts,
                )

                if self.accelerator.is_main_process:
                    for record in gathered_records:
                        for key, value in record["metrics"].items():
                            self.evaluation_metrics.update(key, value)

                        log_batch = self._restore_evaluation_log_batch(record["log_batch"])
                        if log_batch is not None:
                            self._log_batch(global_batch_idx, log_batch, part)
                        global_batch_idx += 1

            if self.accelerator.is_main_process:
                self._log_scalars(self.evaluation_metrics, part)

        for metric in self.metrics:
            metric.to_cpu()

        if was_training:
            self.model.train()

        result = (
            self.evaluation_metrics.result()
            if self.accelerator.is_main_process
            else {}
        )
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        return result

    @staticmethod
    def _make_evaluation_record(batch, batch_metrics):
        """
        Build a small picklable validation record for cross-process gathering.

        Metrics are kept as scalars while only the first generated image is
        transferred as a compressed thumbnail for experiment logging.
        """
        log_batch = {
            key: batch[key]
            for key in ("prompt", "id")
            if key in batch
        }

        generated_images = batch.get("generated")
        if generated_images:
            thumbnail = generated_images[0].resize((256, 256)).convert("RGB")
            image_buffer = BytesIO()
            thumbnail.save(image_buffer, format="JPEG", quality=90)
            log_batch["generated_image"] = image_buffer.getvalue()

        return {
            "metrics": batch_metrics.result(),
            "log_batch": log_batch or None,
        }

    @staticmethod
    def _restore_evaluation_log_batch(log_batch):
        """Restore the compact gathered record expected by ``_log_batch``."""
        if log_batch is None or "generated_image" not in log_batch:
            return None

        restored_batch = {
            key: value
            for key, value in log_batch.items()
            if key != "generated_image"
        }
        with Image.open(BytesIO(log_batch["generated_image"])) as image:
            restored_batch["generated"] = [image.copy()]
        return restored_batch

    def _handle_evaluation_artifacts(self, epoch, part, batch, *, keep):
        """
        Handle full-resolution outputs produced during evaluation.

        ``keep`` is false for samples added by a prepared dataloader to pad the
        final distributed batch. Trainers do not persist evaluation artifacts;
        inferencers override this hook.
        """
        return None


    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.max_grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.model.get_distributed_model().parameters(),
                self.max_grad_norm,
            )

    def _log_grad_norms(self, metric_tracker: MetricTracker):
        """
        Collect gradient norms for logging before gradients are cleared.

        Args:
            metric_tracker (MetricTracker): tracker to update with grad norms.
        """
        if not self.accelerator.sync_gradients:
            return

        grad_norms = self._get_grad_norms()
        for part_name, part_norm in grad_norms.items():
            metric_tracker.update(f"grad_norm/{part_name}", part_norm)

    def _scale_batch_dependent_step(self, step):
        if step is None:
            return None
        return step * self.gradient_accumulation_steps

    def _scale_writer_step(self, step):
        return step // self.gradient_accumulation_steps
            

    @torch.no_grad()
    def _get_grad_norms(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            grad_norms (dict): the calculated norms.
        """
        # ``clip_grad_norm_`` is FSDP2/DTensor-aware. An infinite limit makes
        # it a read-only global norm calculation.
        def compute_params_grad_norm(parameters):
            parameters = [
                parameter
                for parameter in parameters
                if parameter.grad is not None
            ]
            if not parameters:
                return 0
            norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float("inf"),
                norm_type=norm_type,
            )
            if hasattr(norm, "to_local"):
                norm = norm.to_local()
            return norm.item()

        grad_norms = {}
        for group in self.optimizer.param_groups:
            grad_norms[group["name"]] = compute_params_grad_norm(group["params"])

        # Compute total norm
        total_norm = torch.norm(torch.tensor(list(grad_norms.values())), norm_type).item()
        grad_norms["total_norm"] = total_norm

        return grad_norms

    @torch.no_grad()
    def _get_lrs(self):
        """
        Returns lrs for logging.

        Returns:
            lres (dict): last lrs
        """
        lrs = {}
        for last_lr, group in zip(self.lr_scheduler.get_last_lr(), self.optimizer.param_groups):
            lrs[group["name"]] = last_lr

        return lrs

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        raise NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker, part):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{part}/{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        self.accelerator.wait_for_everyone()
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if self.accelerator.is_main_process:
            self.logger.info(f"Saving checkpoint: {filename} ...")

        is_fsdp = self.accelerator.distributed_type == DistributedType.FSDP
        raw_model_state = {}
        model_state_dir = None
        optimizer_state_dir = None

        if is_fsdp and not self.model.checkpoint_trainable_only:
            from accelerate.utils.fsdp_utils import save_fsdp_model

            model_state_dir = f"checkpoint-epoch{epoch}-model"
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model.get_distributed_model(),
                str(self.checkpoint_dir / model_state_dir),
            )
        else:
            raw_model_state = self._get_checkpoint_model_state()

        if is_fsdp:
            from accelerate.utils.fsdp_utils import save_fsdp_optimizer

            optimizer_state_dir = f"checkpoint-epoch{epoch}-optimizer"
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.optimizer,
                self.model.get_distributed_model(),
                str(self.checkpoint_dir / optimizer_state_dir),
            )

        rng_states = gather_object([self._get_rng_state()])
        if self.accelerator.is_main_process:
            state = {
                "arch": type(self.model).__name__,
                "epoch": epoch,
                "optimizer": (
                    None
                    if optimizer_state_dir is not None
                    else self.optimizer.state_dict()
                ),
                "model_state_dir": model_state_dir,
                "optimizer_state_dir": optimizer_state_dir,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": (
                    self.accelerator.scaler.state_dict()
                    if self.accelerator.scaler is not None
                    else None
                ),
                "rng_states": rng_states,
                "config": self.config,
            }
            if model_state_dir is None:
                # Portable model/adapter format used by existing inference.
                state["state_dict"] = self.model.get_state_dict(raw_model_state)
                # Raw keys allow loading into DDP, FSDP1, and FSDP2.
                state["model_state_dict"] = raw_model_state
            torch.save(state, filename)

        self.accelerator.wait_for_everyone()

    @staticmethod
    def _get_rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, rng_states):
        if not rng_states:
            return
        process_index = min(self.accelerator.process_index, len(rng_states) - 1)
        state = rng_states[process_index]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _get_checkpoint_model_state(self):
        """Gather only the state needed to resume the prepared component."""
        is_fsdp = self.accelerator.distributed_type == DistributedType.FSDP
        if not is_fsdp and not self.accelerator.is_main_process:
            return {}

        if self.model.checkpoint_trainable_only:
            # FSDP1 cannot materialize individual FlatParameter entries. Its
            # full-state API is the compatibility fallback; FSDP2 gathers only
            # the selected DTensor-backed adapter parameters.
            if is_fsdp and not self.accelerator.is_fsdp2:
                full_state = self.accelerator.get_state_dict(
                    self.model.get_distributed_model()
                )
                return self.model.select_trainable_model_state_dict(full_state)
            return self.model.get_trainable_model_state_dict()

        full_state = self.accelerator.get_state_dict(
            self.model.get_distributed_model()
        )
        return materialize_state_dict(full_state)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        resume_dir = os.path.dirname(os.path.abspath(resume_path))
        if self.accelerator.is_main_process:
            self.logger.info(f"Loading checkpoint: {resume_path} ...")

        checkpoint = (
            torch.load(resume_path, map_location="cpu", weights_only=False)
            if self.accelerator.is_main_process
            else None
        )
        metadata = None
        if self.accelerator.is_main_process:
            metadata = {
                key: value
                for key, value in checkpoint.items()
                if key not in {"state_dict", "model_state_dict", "optimizer"}
            }
            metadata["has_model_state_dict"] = "model_state_dict" in checkpoint
        metadata_list = [metadata]
        broadcast_object_list(metadata_list)
        metadata = metadata_list[0]
        self.start_epoch = metadata["epoch"] + 1

        # load architecture params from checkpoint.
        if metadata["config"]["model"] != self.config["model"]:
            if self.accelerator.is_main_process:
                self.logger.warning(
                    "Warning: Architecture configuration given in the config file is different from that "
                    "of the checkpoint. This may yield an exception when state_dict is loaded."
                )

        if (
            self.accelerator.is_main_process
            and metadata["has_model_state_dict"]
        ):
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = {}

        if (
            self.accelerator.is_main_process
            and not metadata["has_model_state_dict"]
            and metadata.get("model_state_dir") is None
        ):
            portable_state = checkpoint["state_dict"]
        else:
            portable_state = None

        model_state_dir = metadata.get("model_state_dir")
        if model_state_dir is not None:
            if self.accelerator.distributed_type != DistributedType.FSDP:
                raise RuntimeError(
                    "This checkpoint contains sharded FSDP model state and "
                    "must be resumed with FSDP."
                )
            from accelerate.utils.fsdp_utils import load_fsdp_model

            load_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model.get_distributed_model(),
                os.path.join(resume_dir, model_state_dir),
            )
        elif metadata["has_model_state_dict"]:
            self.model.load_prepared_model_state_dict(
                model_state,
                strict=False,
                broadcast_from_rank0=True,
            )
        else:
            portable_state_list = [portable_state]
            broadcast_object_list(portable_state_list)
            if self.accelerator.distributed_type == DistributedType.FSDP:
                raise RuntimeError(
                    "Legacy portable-only checkpoints cannot be resumed after "
                    "FSDP wrapping. Load them with DDP once and save a new checkpoint."
                )
            self.model.load_state_dict_(portable_state_list[0])

        # FSDP2 optimizer restore may touch the scaler, so initialize it first.
        scaler_state = metadata.get("scaler")
        if scaler_state is not None and self.accelerator.scaler is not None:
            self.accelerator.scaler.load_state_dict(scaler_state)
            if self.accelerator.is_fsdp2:
                self.accelerator.scaler._lazy_init_scale_growth_tracker(
                    self.accelerator.scaler._device
                )

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            metadata["config"]["optimizer"] != self.config["optimizer"]
            or metadata["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            if self.accelerator.is_main_process:
                self.logger.warning(
                    "Warning: Optimizer or lr_scheduler given in the config file is different "
                    "from that of the checkpoint. Optimizer and scheduler parameters "
                    "are not resumed."
                )
        else:
            optimizer_state_dir = metadata.get("optimizer_state_dir")
            if optimizer_state_dir is not None:
                from accelerate.utils.fsdp_utils import load_fsdp_optimizer

                load_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    self.optimizer,
                    self.model.get_distributed_model(),
                    os.path.join(resume_dir, optimizer_state_dir),
                )
            else:
                optimizer_state = (
                    checkpoint["optimizer"]
                    if self.accelerator.is_main_process
                    else None
                )
                optimizer_state_list = [optimizer_state]
                broadcast_object_list(optimizer_state_list)
                self.optimizer.load_state_dict(optimizer_state_list[0])

            self.lr_scheduler.load_state_dict(metadata["lr_scheduler"])

        self._restore_rng_state(metadata.get("rng_states"))

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
            )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            if self.accelerator.is_main_process:
                self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = (
            torch.load(pretrained_path, map_location="cpu", weights_only=False)
            if self.accelerator.is_main_process
            else None
        )

        raw_state = None
        portable_state = None
        model_state_dir = None
        if self.accelerator.is_main_process:
            model_state_dir = checkpoint.get("model_state_dir")
            raw_state = checkpoint.get("model_state_dict")
            portable_state = checkpoint.get("state_dict", checkpoint)

        sharded_state_list = [model_state_dir]
        broadcast_object_list(sharded_state_list)
        model_state_dir = sharded_state_list[0]
        if model_state_dir is not None:
            if self.accelerator.distributed_type != DistributedType.FSDP:
                raise RuntimeError(
                    "This checkpoint contains sharded FSDP model weights and "
                    "must be loaded with an FSDP launch configuration."
                )
            from accelerate.utils.fsdp_utils import load_fsdp_model

            load_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model.get_distributed_model(),
                os.path.join(
                    os.path.dirname(os.path.abspath(pretrained_path)),
                    model_state_dir,
                ),
            )
        else:
            has_raw_state = [raw_state is not None]
            broadcast_object_list(has_raw_state)
            if has_raw_state[0]:
                self.model.load_prepared_model_state_dict(
                    raw_state or {},
                    strict=False,
                    broadcast_from_rank0=True,
                )
            else:
                portable_state_list = [portable_state]
                broadcast_object_list(portable_state_list)
                if self.accelerator.distributed_type == DistributedType.FSDP:
                    raise RuntimeError(
                        "A portable-only checkpoint cannot be loaded after FSDP "
                        "wrapping. Use a checkpoint containing model_state_dict."
                    )
                self.model.load_state_dict_(portable_state_list[0])

        self.accelerator.wait_for_everyone()
           
