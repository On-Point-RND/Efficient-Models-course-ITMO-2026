"""Calibrate the analytical latency/energy model and plot held-out checks.

Run ``python hw1/calibrate.py`` after the Kaggle measurements have been copied
to ``hw1/results``. NumPy, SciPy, and Matplotlib are required locally.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from equations import bytes_moved, energy, flops, latency, memory
from kaggle_measure import BASE_BATCHES, BASE_SIZES


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def check_grid(rows: list[dict[str, str]], metadata: dict) -> None:
    expected = {(s, b) for s in metadata["sizes"] for b in metadata["batches"]}
    observed = [(int(r["S"]), int(r["B"])) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("measurements.csv is incomplete or has duplicate grid points")
    for row in rows:
        s, b = int(row["S"]), int(row["B"])
        should_hold_out = int(s not in BASE_SIZES or b not in BASE_BATCHES)
        if int(row["is_validation"]) != should_hold_out:
            raise ValueError(f"Incorrect train/validation flag at S={s}, B={b}")


def metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        return {"n": 0, "mae": None, "median_relative_error": None, "p95_relative_error": None}
    diff = np.abs(actual[mask] - predicted[mask])
    relative = diff / np.maximum(np.abs(actual[mask]), 1e-12)
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(diff)),
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.percentile(relative, 95)),
    }


def plot_comparison(path: Path, name: str, unit: str, actual: np.ndarray,
                    predicted: np.ndarray, train: np.ndarray, valid: np.ndarray,
                    sizes: list[int], batches: list[int],
                    sizes_array: np.ndarray, batches_array: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    shape = (len(sizes), len(batches))
    # The measurements file is randomized, so map by (S, B) before reshaping.
    order = np.lexsort((batches_array, sizes_array))
    observed_grid = actual[order].reshape(shape)
    predicted_grid = predicted[order].reshape(shape)
    levels = np.r_[observed_grid.ravel(), predicted_grid.ravel()]
    levels = levels[np.isfinite(levels) & (levels > 0)]
    norm = LogNorm(vmin=float(np.min(levels)), vmax=float(np.max(levels)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained")
    for ax, grid, title in zip(axes[:2], (observed_grid, predicted_grid),
                               ("Measured", "Analytical prediction")):
        image = ax.imshow(grid, origin="lower", aspect="auto", norm=norm, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(batches)), batches, rotation=55, ha="right")
        ax.set_yticks(range(len(sizes)), sizes)
        ax.set_xlabel("Batch size B")
        ax.set_ylabel("Image size S [pixels]")
    fig.colorbar(image, ax=axes[:2], shrink=0.82, label=f"{name} [{unit}]")

    ax = axes[2]
    ax.scatter(predicted[train], actual[train], s=25, alpha=0.75, label="Calibration grid")
    ax.scatter(predicted[valid], actual[valid], s=30, alpha=0.8, label="Held out")
    lo = float(np.min(levels))
    hi = float(np.max(levels))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Exact agreement")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo * 0.85, hi * 1.2)
    ax.set_ylim(lo * 0.85, hi * 1.2)
    ax.set_xlabel(f"Analytical prediction [{unit}]")
    ax.set_ylabel(f"Measured [{unit}]")
    ax.set_title("All 132 grid points")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="upper left")
    fig.suptitle(f"{name}: complete Kaggle T4 grid")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def calibrate(results: Path) -> dict:
    rows = read_rows(results / "measurements.csv")
    metadata = json.loads((results / "metadata.json").read_text())
    check_grid(rows, metadata)
    s = np.array([int(r["S"]) for r in rows], dtype=float)
    b = np.array([int(r["B"]) for r in rows], dtype=float)
    clean = np.array([r["status"] == "ok" and not r["measurement_warning"] for r in rows])
    train = clean & np.array([r["is_validation"] == "0" for r in rows])
    valid = clean & ~np.array([r["is_validation"] == "0" for r in rows])
    if train.sum() < 10 or valid.sum() < 1:
        raise ValueError("Need at least 10 clean calibration points and one held-out point")
    observed_t = np.array([float(r["latency_s"]) if r["latency_s"] else np.nan for r in rows])
    observed_m = np.array([float(r["peak_memory_bytes"]) if r["peak_memory_bytes"] else np.nan for r in rows])
    observed_e = np.array([float(r["energy_j"]) if r["energy_j"] else np.nan for r in rows])
    if not all(np.isfinite(x[clean]).all() and (x[clean] > 0).all()
               for x in (observed_t, observed_m, observed_e)):
        raise ValueError("Clean measurements must be finite and positive")

    f_gflop = flops(s, b) / 1e9
    d_gb = bytes_moved(s, b) / 1e9
    observed_ms = observed_t * 1_000

    def predicted_ms(parameters: np.ndarray) -> np.ndarray:
        launch_ms, compute_tflops, bandwidth_tbps = parameters
        return launch_ms + np.maximum(f_gflop / compute_tflops, d_gb / bandwidth_tbps)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return np.log(predicted_ms(parameters)[train]) - np.log(observed_ms[train])

    fits = [least_squares(residual, initial,
                          bounds=([0.0, 0.01, 0.001], [10.0, 20.0, 5.0]),
                          max_nfev=5_000)
            for initial in ([0.4, 2.0, 0.1], [0.1, 4.0, 0.02], [1.0, 1.0, 0.3])]
    best = min(fits, key=lambda result: result.cost)
    if not best.success:
        raise RuntimeError(f"Latency calibration failed: {best.message}")
    launch_ms, compute_tflops, bandwidth_tbps = map(float, best.x)
    theta = {
        "launch_ms": launch_ms,
        "compute_tflops": compute_tflops,
        "bandwidth_tbps": bandwidth_tbps,
    }
    predicted_t = latency(s, b, theta)
    # Log-domain least squares for E = P * predicted_latency has a closed form.
    effective_power = float(np.exp(np.mean(np.log(observed_e[train] / predicted_t[train]))))
    theta["gpu_power_w"] = effective_power
    predicted_e = energy(s, b, theta)
    predicted_m = memory(s, b)

    compute_ms = f_gflop / compute_tflops
    bandwidth_ms = d_gb / bandwidth_tbps
    regimes = np.where(launch_ms >= np.maximum(compute_ms, bandwidth_ms), "launch",
                       np.where(compute_ms >= bandwidth_ms, "compute", "memory"))
    base_s = np.isin(s, BASE_SIZES)
    base_b = np.isin(b, BASE_BATCHES)
    groups = {
        "unseen_size_only": valid & ~base_s & base_b,
        "unseen_batch_only": valid & base_s & ~base_b,
        "both_unseen": valid & ~base_s & ~base_b,
    }

    # Errors for latency and energy are evaluated from predictions, never from
    # measured latency on the held-out set.
    summary = {
        "gpu": metadata["gpu"],
        "total_points": len(rows),
        "calibration_points": int(train.sum()),
        "validation_points": int(valid.sum()),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "regime_counts": dict(Counter(regimes.tolist())),
        "objective": "sum of squared log(predicted/observed latency) on clean base-grid points",
        "latency_ms": {
            "train": metrics(observed_ms, predicted_t * 1_000, train),
            "validation": metrics(observed_ms, predicted_t * 1_000, valid),
        },
        "peak_memory_mib": {
            "train": metrics(observed_m / 2**20, predicted_m / 2**20, train),
            "validation": metrics(observed_m / 2**20, predicted_m / 2**20, valid),
        },
        "energy_mj": {
            "train": metrics(observed_e * 1_000, predicted_e * 1_000, train),
            "validation": metrics(observed_e * 1_000, predicted_e * 1_000, valid),
        },
        "validation_groups": {
            key: {
                "latency": metrics(observed_ms, predicted_t * 1_000, mask),
                "memory": metrics(observed_m / 2**20, predicted_m / 2**20, mask),
                "energy": metrics(observed_e * 1_000, predicted_e * 1_000, mask),
            }
            for key, mask in groups.items()
        },
        "memory_measured_to_ideal_ratio": {
            "minimum": float(np.min(observed_m[clean] / predicted_m[clean])),
            "median": float(np.median(observed_m[clean] / predicted_m[clean])),
            "maximum": float(np.max(observed_m[clean] / predicted_m[clean])),
        },
    }
    (results / "theta.json").write_text(json.dumps(theta, indent=2) + "\n")
    (results / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = [
        "S", "B", "is_validation", "status", "regime",
        "measured_latency_s", "predicted_latency_s",
        "measured_peak_memory_bytes", "predicted_peak_memory_bytes",
        "measured_energy_j", "predicted_energy_j", "flops", "nominal_bytes_moved",
    ]
    with (results / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "S": int(s[i]), "B": int(b[i]),
                "is_validation": row["is_validation"], "status": row["status"],
                "regime": regimes[i],
                "measured_latency_s": row["latency_s"],
                "predicted_latency_s": float(predicted_t[i]),
                "measured_peak_memory_bytes": row["peak_memory_bytes"],
                "predicted_peak_memory_bytes": float(predicted_m[i]),
                "measured_energy_j": row["energy_j"],
                "predicted_energy_j": float(predicted_e[i]),
                "flops": float(flops(s[i], b[i])),
                "nominal_bytes_moved": float(bytes_moved(s[i], b[i])),
            })

    figure_dir = results / "figures"
    figure_dir.mkdir(exist_ok=True)
    for name, unit, actual, predicted in (
        ("Latency", "ms", observed_ms, predicted_t * 1_000),
        ("Peak allocated memory", "MiB", observed_m / 2**20, predicted_m / 2**20),
        ("Energy", "mJ", observed_e * 1_000, predicted_e * 1_000),
    ):
        filename = "analytic_" + {"Latency": "latency", "Peak allocated memory": "memory", "Energy": "energy"}[name] + ".png"
        plot_comparison(figure_dir / filename, name, unit, actual, predicted,
                        train, valid, metadata["sizes"], metadata["batches"], s, b)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    summary = calibrate(args.results_dir)
    print(json.dumps({
        "theta": json.loads((args.results_dir / "theta.json").read_text()),
        "validation": {name: summary[name]["validation"]
                       for name in ("latency_ms", "peak_memory_mib", "energy_mj")},
    }, indent=2))


if __name__ == "__main__":
    main()
