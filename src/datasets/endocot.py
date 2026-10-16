from __future__ import annotations

import csv
import random
from pathlib import Path

from src.datasets.base_dataset import BaseDataset


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


class EndoCoTDataset(BaseDataset):
    """
    Image-edit pairs from internlm/EndoCoT-Data.

    Each row trains FLUX.2 Klein to edit ``edit_image`` into ``image`` under
    ``prompt``. Validation keeps held-out episodes out of training.

    ``pair_mode``:
      - ``all``: every CSV row (including intermediate CoT steps)
      - ``final``: only start-image -> final-solution pairs per episode
    """

    def __init__(
        self,
        data_root: str,
        tasks: list[str],
        split: str,
        metadata_name: str = "metadata_edit.csv",
        pair_mode: str = "all",
        val_episodes_per_task: int = 8,
        max_val_samples: int = 16,
        seed: int = 0,
        *args,
        **kwargs,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        if pair_mode not in {"all", "final"}:
            raise ValueError(f"pair_mode must be 'all' or 'final', got {pair_mode}")
        self.data_root = Path(data_root)
        self.split = split
        self.metadata_name = metadata_name
        self.pair_mode = pair_mode
        self.val_episodes_per_task = val_episodes_per_task
        self.max_val_samples = max_val_samples
        self.seed = seed
        index = self._build_index(tasks)
        super().__init__(index, *args, **kwargs)
        if len(self._index) == 0:
            raise RuntimeError(
                f"EndoCoT {split} split is empty. Expected extracted images under {self.data_root}."
            )

    @staticmethod
    def _episode_final_rows(rows: list[dict]) -> list[dict]:
        finals = [row for row in rows if row["final"]]
        return finals or rows[-1:]

    def _build_index(self, tasks: list[str]) -> list[dict]:
        grouped: dict[str, dict[str, list[dict]]] = {}
        for task in tasks:
            csv_path = self.data_root / task / self.metadata_name
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing EndoCoT metadata: {csv_path}")
            grouped[task] = self._rows_by_episode(task, csv_path)

        index = []
        for task, episodes in grouped.items():
            episode_ids = sorted(episodes)
            rng = random.Random(self.seed)
            rng.shuffle(episode_ids)
            held_out = set(episode_ids[: min(self.val_episodes_per_task, len(episode_ids))])
            selected_ids = held_out if self.split == "val" else [item for item in episode_ids if item not in held_out]
            if isinstance(selected_ids, set):
                selected_ids = sorted(selected_ids)
            task_rows = []
            for episode_id in selected_ids:
                rows = episodes[episode_id]
                if self.split == "val" or self.pair_mode == "final":
                    task_rows.extend(self._episode_final_rows(rows))
                else:
                    task_rows.extend(rows)
            if self.split == "val":
                task_rows = task_rows[: self.max_val_samples]
            index.extend(task_rows)
        return index

    def _rows_by_episode(self, task: str, csv_path: Path) -> dict[str, list[dict]]:
        episodes: dict[str, list[dict]] = {}
        with csv_path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader([handle.readline()]))
            columns = {name: index for index, name in enumerate(header)}
            for required in ("image", "edit_image", "prompt"):
                if required not in columns:
                    raise ValueError(f"{csv_path} is missing the {required} column.")
            image_index = columns["image"]
            edit_index = columns["edit_image"]
            final_index = columns.get("final_image")
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = next(csv.reader([line]))
                episode = row[edit_index] or row[image_index]
                image_path = row[image_index].replace("\\", "/")
                # metadata_edit uses bools; metadata_edit_final stores a step
                # count in final_image while image already points at solution.png.
                is_final = "/final/" in image_path
                if not is_final and final_index is not None:
                    is_final = _as_bool(row[final_index])
                episodes.setdefault(episode, []).append(
                    {
                        "task": task,
                        "csv_path": str(csv_path),
                        "offset": offset,
                        "final": is_final,
                    }
                )
        return episodes

    def _read_csv_row(self, csv_path: str, offset: int) -> dict[str, str]:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            header = handle.readline()
            handle.seek(offset)
            line = handle.readline()
        names = next(csv.reader([header]))
        values = next(csv.reader([line]))
        return dict(zip(names, values))

    def _resolve(self, task: str, relative_path: str) -> Path:
        relative = Path(relative_path)
        candidates = [
            self.data_root / task / relative,
            self.data_root / relative,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not find {relative_path} under {self.data_root / task}. "
            "Extract the task archive before training."
        )

    def __getitem__(self, ind):
        meta = self._index[ind]
        row = self._read_csv_row(meta["csv_path"], meta["offset"])
        instance_data = {
            "pixel_values": self.load_object(self._resolve(meta["task"], row["image"])),
            "cond_pixel_values": self.load_object(self._resolve(meta["task"], row["edit_image"])),
            "prompt": row["prompt"],
            "task": meta["task"],
        }
        return self.preprocess_data(instance_data)
