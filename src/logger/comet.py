from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import comet_ml


class CometMLWriter:
    """
    Class for experiment tracking via Comet ML.

    See https://www.comet.com/docs/ for details.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        API key is expected to be provided by the user (e.g. COMET_API_KEY env var
        or via `comet login` / `comet_ml.login()`).

        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            entity (str | None): Comet workspace.
            run_id (str | None): Comet experiment_key.
            run_name (str | None): Comet experiment name.
            mode (str): "online" or "offline".
        """
        self.logger = logger
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

        self.experiment = None
        self.run_id = run_id

        # online/offline switch
        online = (mode or "online").lower() != "offline"


        experiment_config_kwargs = {
            "log_code": False,
            "log_graph": False,
            "parse_args": False,
            "log_git_metadata": False,
            "log_git_patch": False,
            "log_env_details": False,
            "display_summary_level": 0,
            "name": run_name,
        }

        allowed_cfg_keys = {
            "disabled",
            "log_code",
            "log_graph",
            "parse_args",
            "display_summary_level",
            "log_git_metadata",
            "log_git_patch",
            "log_env_details",
            "log_env_gpu",
            "log_env_host",
            "log_env_cpu",
            "log_env_network",
            "log_env_disk",
            "auto_output_logging",
            "auto_param_logging",
            "auto_metric_logging",
            "auto_log_co2",
            "auto_metric_step_rate",
            "auto_histogram_epoch_rate",
            "auto_histogram_gradient_logging",
            "auto_histogram_activation_logging",
            "auto_histogram_tensorboard_logging",
            "auto_histogram_weight_logging",
            "offline_directory",
            "distributed_node_identifier",
            "name",
            "tags",
        }
        for k, v in list(kwargs.items()):
            if k in allowed_cfg_keys:
                experiment_config_kwargs[k] = v

        experiment_config = comet_ml.ExperimentConfig(**experiment_config_kwargs)

        if online:
            try:
                comet_ml.login()
            except Exception as e:
                self.logger.warning(f"comet_ml.login() failed (continuing): {e}")

        self.experiment = comet_ml.start(
            project_name=project_name,
            workspace=entity,
            experiment_key=self.run_id,
            online=online,
            experiment_config=experiment_config,
        )

        try:
            if self.run_id is None:
                self.run_id = self.experiment.get_key()
        except Exception:
            pass

        try:
            self.experiment.log_parameters(self._to_jsonable(project_config))
        except Exception as e:
            self.logger.warning(f"Failed to log project_config to Comet parameters: {e}")


    def _to_jsonable(self, obj: Any) -> Any:
        """
        Best-effort conversion to JSON-encodable types (Comet parameters/metadata need this).
        """
        # numpy scalars
        if isinstance(obj, (np.generic,)):
            return obj.item()

        # pandas / numpy
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()

        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        # dict / list / tuple
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]

        if hasattr(obj, "detach") and callable(getattr(obj, "detach")):
            try:
                return obj.detach().cpu().numpy().tolist()
            except Exception:
                return str(obj)

        # fallback
        try:
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            return str(obj)
        except Exception:
            return repr(obj)


    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.
        """
        self.mode = mode
        previous_step = self.step
        self.step = step


        if self.experiment is not None:
            try:
                self.experiment.set_step(self.step)
            except Exception:
                pass

        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "general/steps_per_sec",
                (self.step - previous_step) / max(duration.total_seconds(), 1e-12),
            )
            self.timer = datetime.now()

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.
        """
        self.logger.info(f"Step {self.step}: {scalar_name} = {scalar}")
        if self.experiment is None:
            return

        self.experiment.log_metric(name=scalar_name, value=scalar, step=self.step)

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.
        """
        for scalar_name, scalar_value in scalars.items():
            self.logger.info(f"Step {self.step}: {scalar_name} = {scalar_value}")

        if self.experiment is None:
            return

        self.experiment.log_metrics(scalars, step=self.step)

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.
        """
        if self.experiment is None:
            return


        self.experiment.log_image(image_data=image, name=image_name, step=self.step)

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log an audio to the experiment tracker.
        """
        if self.experiment is None:
            return

        if hasattr(audio, "detach") and callable(getattr(audio, "detach")):
            audio_np = audio.detach().cpu().numpy().T
        else:
            audio_np = np.asarray(audio)

        self.experiment.log_audio(
            audio_data=audio_np,
            sample_rate=sample_rate,
            file_name=audio_name,
            step=self.step,
        )

    def add_text(self, text_name, text):
        """
        Log text to the experiment tracker.
        """
        if self.experiment is None:
            return

        self.experiment.log_text(
            text=str(text),
            step=self.step,
            metadata={"name": text_name, "format": "html"},
        )

    def add_file(self, file_path):
        """Upload a file as a Comet experiment asset."""
        if self.experiment is None:
            return

        file_path = Path(file_path)
        self.experiment.log_asset(
            file_data=str(file_path),
            file_name=file_path.name,
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram to the experiment tracker.
        """
        if self.experiment is None:
            return

        if hasattr(values_for_hist, "detach") and callable(getattr(values_for_hist, "detach")):
            values_np = values_for_hist.detach().cpu().numpy()
        else:
            values_np = np.asarray(values_for_hist)

        try:
            if bins is not None:
                self.experiment.log_histogram_3d(values=values_np, name=hist_name, step=self.step, bins=bins)
            else:
                self.experiment.log_histogram_3d(values=values_np, name=hist_name, step=self.step)
        except TypeError:
            self.experiment.log_histogram_3d(values=values_np, name=hist_name, step=self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log table to the experiment tracker.
        """
        if self.experiment is None:
            return

        filename = str(table_name)
        if not any(filename.endswith(ext) for ext in (".csv", ".tsv", ".json", ".md", ".html")):
            filename = f"{filename}.csv"

        self.experiment.log_table(filename=filename, tabular_data=table)

    def add_images(self, images_name, images):
        """
        Log multiple images
        """
        if self.experiment is None:
            return

        for i, image in enumerate(images):
            name = f"{images_name}/{i}"
            self.experiment.log_image(image_data=image, name=name, step=self.step)
