#!/usr/bin/env python3
"""
Generate train and eval metric plots for a Hugging Face SFT run.

The script reads Hugging Face Trainer `trainer_state.json` files, including the
latest checkpoint while a run is still in progress, and writes PNG plots plus a
CSV export of the metric history.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train and eval metrics from a Hugging Face trainer_state.json file."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory that contains trainer_state.json or checkpoint-* subdirectories.",
    )
    parser.add_argument(
        "--trainer-state",
        help="Optional explicit path to trainer_state.json. Defaults to the run root, then the latest checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where plots and CSV exports will be written. Defaults to <run-dir>/plots.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Trailing moving-average window for the plotted training loss and perplexity.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    return resolve_repo_path(value)


def resolve_run_dir(path: str | Path) -> Path:
    run_dir = resolve_path(path)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory is not a directory: {run_dir}")
    return run_dir


def resolve_trainer_state_path(
    run_dir: Path, trainer_state_override: str | None
) -> Path:
    if trainer_state_override:
        trainer_state_path = resolve_path(trainer_state_override)
        if not trainer_state_path.exists():
            raise FileNotFoundError(
                f"trainer_state.json does not exist: {trainer_state_path}"
            )
        return trainer_state_path

    root_trainer_state = run_dir / "trainer_state.json"
    if root_trainer_state.exists():
        return root_trainer_state

    latest_checkpoint_state = find_latest_checkpoint_trainer_state(run_dir)
    if latest_checkpoint_state is not None:
        return latest_checkpoint_state

    raise FileNotFoundError(
        "Could not find trainer_state.json in the run directory or checkpoint-* subdirectories."
    )


def find_latest_checkpoint_trainer_state(run_dir: Path) -> Path | None:
    checkpoint_states: list[tuple[int, Path]] = []
    for checkpoint_dir in run_dir.glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        try:
            step = int(checkpoint_dir.name.split("-", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if trainer_state_path.exists():
            checkpoint_states.append((step, trainer_state_path))

    if not checkpoint_states:
        return None
    checkpoint_states.sort()
    return checkpoint_states[-1][1]


def load_log_history(trainer_state_path: Path) -> list[dict[str, Any]]:
    with trainer_state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    log_history = payload.get("log_history", [])
    if not isinstance(log_history, list):
        raise ValueError(
            f"`log_history` must be a list in {trainer_state_path}, got {type(log_history)}."
        )
    return [entry for entry in log_history if isinstance(entry, dict)]


def get_step(entry: dict[str, Any], fallback: int) -> int:
    step = entry.get("step", entry.get("global_step", fallback))
    try:
        return int(step)
    except (TypeError, ValueError):
        return fallback


def get_epoch(entry: dict[str, Any]) -> float | None:
    value = entry.get("epoch")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_float(entry: dict[str, Any], key: str) -> float | None:
    value = entry.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_perplexity(loss_value: float | None) -> float | None:
    if loss_value is None:
        return None
    try:
        return math.exp(loss_value)
    except OverflowError:
        return None


def extract_metric_rows(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(log_history, start=1):
        step = get_step(entry, index)
        epoch = get_epoch(entry)
        train_loss = get_float(entry, "loss")
        eval_loss = get_float(entry, "eval_loss")
        learning_rate = get_float(entry, "learning_rate")
        grad_norm = get_float(entry, "grad_norm")

        if train_loss is not None:
            rows.append(
                {
                    "split": "train",
                    "step": step,
                    "epoch": epoch,
                    "loss": train_loss,
                    "perplexity": safe_perplexity(train_loss),
                    "learning_rate": learning_rate,
                    "grad_norm": grad_norm,
                }
            )

        if eval_loss is not None:
            rows.append(
                {
                    "split": "eval",
                    "step": step,
                    "epoch": epoch,
                    "loss": eval_loss,
                    "perplexity": safe_perplexity(eval_loss),
                    "learning_rate": learning_rate,
                    "grad_norm": None,
                }
            )
    return rows


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)

    smoothed: list[float] = []
    running_total = 0.0
    for index, value in enumerate(values):
        running_total += value
        if index >= window:
            running_total -= values[index - window]
        start = max(0, index - window + 1)
        count = index - start + 1
        smoothed.append(running_total / count)
    return smoothed


def values_for_key(
    rows: list[dict[str, Any]], key: str
) -> tuple[list[int], list[float], list[float | None]]:
    steps: list[int] = []
    values: list[float] = []
    epochs: list[float | None] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        steps.append(int(row["step"]))
        values.append(float(value))
        epochs.append(row.get("epoch"))
    return steps, values, epochs


def annotate_best_eval(ax: Any, eval_rows: list[dict[str, Any]], key: str) -> None:
    best_candidates = [row for row in eval_rows if row.get(key) is not None]
    if not best_candidates:
        return
    best = min(best_candidates, key=lambda row: float(row[key]))
    ax.scatter(
        [best["step"]],
        [float(best[key])],
        color="tab:red",
        marker="*",
        s=180,
        zorder=5,
        label=f"best eval @ step {best['step']}",
    )


def save_loss_plot(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    output_path: Path,
    smoothing_window: int,
) -> None:
    train_steps, train_losses, _ = values_for_key(train_rows, "loss")
    eval_steps, eval_losses, _ = values_for_key(eval_rows, "loss")

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_losses:
        ax.plot(
            train_steps,
            train_losses,
            color="tab:blue",
            alpha=0.25,
            linewidth=1.5,
            label="train loss (raw)",
        )
        ax.plot(
            train_steps,
            moving_average(train_losses, smoothing_window),
            color="tab:blue",
            linewidth=2.0,
            label=f"train loss (avg {max(smoothing_window, 1)})",
        )
    if eval_losses:
        ax.plot(
            eval_steps,
            eval_losses,
            color="tab:orange",
            marker="o",
            linewidth=2.0,
            label="eval loss",
        )
        annotate_best_eval(ax, eval_rows, "loss")

    ax.set_title("SFT Loss Curves")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_perplexity_plot(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    output_path: Path,
    smoothing_window: int,
) -> None:
    train_steps, train_perplexities, _ = values_for_key(train_rows, "perplexity")
    eval_steps, eval_perplexities, _ = values_for_key(eval_rows, "perplexity")

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_perplexities:
        ax.plot(
            train_steps,
            train_perplexities,
            color="tab:green",
            alpha=0.25,
            linewidth=1.5,
            label="train perplexity (raw)",
        )
        ax.plot(
            train_steps,
            moving_average(train_perplexities, smoothing_window),
            color="tab:green",
            linewidth=2.0,
            label=f"train perplexity (avg {max(smoothing_window, 1)})",
        )
    if eval_perplexities:
        ax.plot(
            eval_steps,
            eval_perplexities,
            color="tab:red",
            marker="o",
            linewidth=2.0,
            label="eval perplexity",
        )
        annotate_best_eval(ax, eval_rows, "perplexity")

    ax.set_title("SFT Perplexity Curves")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Perplexity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_learning_rate_plot(
    train_rows: list[dict[str, Any]], output_path: Path
) -> bool:
    steps, learning_rates, _ = values_for_key(train_rows, "learning_rate")
    if not learning_rates:
        return False

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        steps,
        learning_rates,
        color="tab:purple",
        linewidth=2.0,
        label="learning rate",
    )
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def save_metric_history_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "split",
        "step",
        "epoch",
        "loss",
        "perplexity",
        "learning_rate",
        "grad_norm",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def describe_best_eval(eval_rows: list[dict[str, Any]]) -> str:
    best_candidates = [row for row in eval_rows if row.get("loss") is not None]
    if not best_candidates:
        return "No eval checkpoints found in trainer_state.json."
    best = min(best_candidates, key=lambda row: float(row["loss"]))
    epoch = best.get("epoch")
    epoch_suffix = ""
    if epoch is not None:
        epoch_suffix = f", epoch {epoch:.3f}"
    return (
        f"Best eval loss {float(best['loss']):.6f} "
        f"(perplexity {float(best['perplexity']):.6f}) at step {best['step']}{epoch_suffix}."
    )


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    trainer_state_path = resolve_trainer_state_path(run_dir, args.trainer_state)
    output_dir = resolve_path(args.output_dir) if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_history = load_log_history(trainer_state_path)
    rows = extract_metric_rows(log_history)
    if not rows:
        raise ValueError(f"No train or eval metrics found in {trainer_state_path}.")

    train_rows = split_rows(rows, "train")
    eval_rows = split_rows(rows, "eval")

    loss_plot_path = output_dir / "loss_curve.png"
    perplexity_plot_path = output_dir / "perplexity_curve.png"
    learning_rate_plot_path = output_dir / "learning_rate_curve.png"
    csv_path = output_dir / "metric_history.csv"

    save_loss_plot(train_rows, eval_rows, loss_plot_path, args.smoothing_window)
    save_perplexity_plot(
        train_rows, eval_rows, perplexity_plot_path, args.smoothing_window
    )
    has_learning_rate = save_learning_rate_plot(train_rows, learning_rate_plot_path)
    save_metric_history_csv(rows, csv_path)

    print(f"Using trainer state: {trainer_state_path}")
    print(f"Saved: {loss_plot_path}")
    print(f"Saved: {perplexity_plot_path}")
    if has_learning_rate:
        print(f"Saved: {learning_rate_plot_path}")
    else:
        print("Skipped learning rate plot because no learning rate values were logged.")
    print(f"Saved: {csv_path}")
    print(describe_best_eval(eval_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
