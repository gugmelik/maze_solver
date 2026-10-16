from __future__ import annotations

from pathlib import Path

from src.datasets.base_dataset import BaseDataset


MAZE_PROMPT = (
    "Draw a continuous red line starting from the yellow dot towards the blue "
    "dot step-by-step, avoiding all walls. Mark the current end of the path "
    "with a green dot. Direction keys: D=Down, U=Up, R=Right, L=Left."
)


class EndoCoTMazeTestDataset(BaseDataset):
    """
    Official EndoCoT Maze test folders (`8_test`, `16_test`, `32_test`).

    Each sample is a start maze image with an optional ground-truth solution
    used only for IoU accuracy logging.
    """

    def __init__(
        self,
        data_root: str,
        level: int,
        *args,
        **kwargs,
    ):
        self.data_root = Path(data_root)
        self.level = int(level)
        self.test_dir = self.data_root / f"{self.level}_test"
        if not self.test_dir.is_dir():
            raise FileNotFoundError(
                f"Missing EndoCoT Maze test directory: {self.test_dir}. "
                "Download Maze/{8,16,32}_test from InternLM/EndoCoT first."
            )
        index = self._build_index()
        super().__init__(index, *args, **kwargs)
        if len(self._index) == 0:
            raise RuntimeError(f"No Maze test inputs found under {self.test_dir}")

    def _build_index(self) -> list[dict]:
        index = []
        for image_path in sorted(self.test_dir.glob("*.png")):
            if image_path.name.endswith("_solution.png"):
                continue
            sample_id = image_path.stem
            solution_path = self.test_dir / f"{sample_id}_solution.png"
            index.append(
                {
                    "id": sample_id,
                    "task": "Maze",
                    "level": self.level,
                    "image_path": str(image_path),
                    "solution_path": str(solution_path) if solution_path.is_file() else None,
                    "prompt": MAZE_PROMPT,
                }
            )
        return index

    def __getitem__(self, ind):
        meta = self._index[ind]
        condition = self.load_object(meta["image_path"])
        target = (
            self.load_object(meta["solution_path"])
            if meta["solution_path"] is not None
            else condition
        )
        instance_data = {
            "id": meta["id"],
            "task": meta["task"],
            "level": meta["level"],
            "prompt": meta["prompt"],
            "cond_pixel_values": condition,
            "pixel_values": target,
        }
        return self.preprocess_data(instance_data)
