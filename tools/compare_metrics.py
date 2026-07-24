#!/usr/bin/env python3
"""Plot validation metrics from one or more saved training runs."""

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
DEFAULT_SAVED_DIR = ROOT_PATH / "saved"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare validation metrics stored in "
            "saved/<run>/metrics/validation.json."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run names under saved/ or paths to run directories.",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric name such as db_liv_small/dino_score.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        help=(
            "Checkpoint epoch numbers such as 1 5 10. "
            "By default, plot every validation iteration."
        ),
    )
    parser.add_argument(
        "--saved-dir",
        type=Path,
        default=DEFAULT_SAVED_DIR,
        help=f"Directory containing runs (default: {DEFAULT_SAVED_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. By default, create a temporary PNG.",
    )
    return parser.parse_args()


def resolve_run_dir(run, saved_dir):
    run_path = Path(run).expanduser()
    if run_path.is_dir():
        return run_path.resolve()

    run_path = saved_dir.expanduser().resolve() / run
    if not run_path.is_dir():
        raise ValueError(f"Run directory does not exist: {run_path}")
    return run_path


def load_validation_metrics(run_dir):
    metrics_path = run_dir / "metrics" / "validation.json"
    if not metrics_path.is_file():
        raise ValueError(f"Validation metrics do not exist: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    result = {}
    for raw_iteration, raw_metrics in payload.items():
        try:
            iteration = int(raw_iteration)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid iteration key {raw_iteration!r} in {metrics_path}"
            ) from error

        if not isinstance(raw_metrics, dict):
            raise ValueError(
                f"Metrics at iteration {iteration} must be an object"
            )

        result[iteration] = {}
        for metric_name, value in raw_metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Metric {metric_name!r} at iteration {iteration} "
                    f"is not numeric: {value!r}"
                )
            result[iteration][metric_name] = float(value)

    if not result:
        raise ValueError(f"Validation metrics are empty: {metrics_path}")
    return result


def load_epoch_len(run_dir):
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"Run config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    epoch_len = config.get("trainer", {}).get("epoch_len")
    if epoch_len is None:
        raise ValueError(
            f"trainer.epoch_len is null in {config_path}; "
            "select iterations instead of checkpoints"
        )
    try:
        return int(epoch_len)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"trainer.epoch_len must be an integer in {config_path}"
        ) from error


def checkpoint_exists(run_dir, checkpoint_name):
    return (
        (run_dir / "checkpoints" / checkpoint_name).is_file()
        or (run_dir / checkpoint_name).is_file()
    )


def select_points(run_dir, validation_metrics, checkpoints):
    if not checkpoints:
        iterations = sorted(validation_metrics)
        return [
            {
                "label": str(iteration),
                "iteration": iteration,
                "checkpoint_epoch": None,
                "metrics": validation_metrics[iteration],
            }
            for iteration in iterations
        ]

    epoch_len = load_epoch_len(run_dir)
    points = []
    for epoch in checkpoints:
        if epoch < 0:
            raise ValueError(
                f"Checkpoint epoch must be non-negative, got {epoch}"
            )
        checkpoint_name = f"checkpoint-epoch{epoch}.pth"
        if epoch > 0 and not checkpoint_exists(run_dir, checkpoint_name):
            raise ValueError(
                f"Checkpoint does not exist for run {run_dir.name}: "
                f"{checkpoint_name}"
            )

        iteration = epoch * epoch_len
        if iteration not in validation_metrics:
            raise ValueError(
                f"No validation metrics for {run_dir.name} at iteration "
                f"{iteration} ({checkpoint_name})"
            )

        points.append(
            {
                "label": str(epoch),
                "iteration": iteration,
                "checkpoint_epoch": epoch,
                "metrics": validation_metrics[iteration],
            }
        )
    return points


def collect_runs(args):
    runs = []
    for run in args.runs:
        run_dir = resolve_run_dir(run, args.saved_dir)
        validation_metrics = load_validation_metrics(run_dir)
        points = select_points(
            run_dir,
            validation_metrics,
            args.checkpoints,
        )
        runs.append(
            {
                "name": run_dir.name,
                "path": run_dir,
                "points": points,
            }
        )
    return runs


def validate_metric(runs, metric):
    available_metrics = {
        metric_name
        for run in runs
        for point in run["points"]
        for metric_name in point["metrics"]
    }
    if metric not in available_metrics:
        raise ValueError(f"Requested metric is unavailable: {metric}")


def print_values(runs, metric):
    print("run\tpoint\titeration\tmetric\tvalue")
    for run in runs:
        for point in run["points"]:
            value = point["metrics"].get(metric)
            rendered_value = "NA" if value is None else f"{value:.8g}"
            print(
                f"{run['name']}\t{point['label']}\t"
                f"{point['iteration']}\t{metric}\t{rendered_value}"
            )


def default_output_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.gettempdir()) / f"metrics_comparison_{timestamp}.png"


def plot_metric(runs, metric, output_path):
    figure, axis = plt.subplots(
        figsize=(max(8, 1.4 * max(len(run["points"]) for run in runs)), 5),
    )

    all_labels = []
    use_checkpoint_axis = all(
        point["checkpoint_epoch"] is not None
        for run in runs
        for point in run["points"]
    )
    for run in runs:
        labels = [point["label"] for point in run["points"]]
        values = [
            point["metrics"].get(metric, math.nan)
            for point in run["points"]
        ]
        x_values = (
            list(range(len(labels)))
            if use_checkpoint_axis
            else [point["iteration"] for point in run["points"]]
        )
        axis.plot(
            x_values,
            values,
            marker="o",
            linewidth=2,
            label=run["name"],
        )
        if len(labels) > len(all_labels):
            all_labels = labels

    axis.set_title(metric)
    axis.set_xlabel(
        "checkpoint epoch" if use_checkpoint_axis else "iteration"
    )
    axis.set_ylabel("value")
    if use_checkpoint_axis:
        axis.set_xticks(
            range(len(all_labels)),
            all_labels,
            rotation=0,
            ha="center",
        )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    args = parse_args()
    try:
        runs = collect_runs(args)
        validate_metric(runs, args.metric)
        print_values(runs, args.metric)
        output_path = plot_metric(
            runs,
            args.metric,
            args.output or default_output_path(),
        )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"\nPlot saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
