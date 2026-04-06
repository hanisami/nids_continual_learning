from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "CGAN diagnostics requires scikit-learn. Install dependencies from requirements.txt."
    ) from exc


def to_numpy(x: Any) -> np.ndarray:
    """Convert tensors/arrays/dataframes into a detached NumPy array."""
    if x is None:
        return np.asarray([])
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    return np.asarray(x)


def sample_rows(x: Any, n: int, random_state: int = 42) -> np.ndarray:
    """Randomly sample up to n rows without replacement."""
    arr = to_numpy(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] <= n:
        return arr
    rng = np.random.default_rng(random_state)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[idx]


def _random_pairwise_distance_mean(
    x: np.ndarray,
    n_pairs: int,
    random_state: int,
) -> float:
    """Estimate the mean pairwise Euclidean distance via random unique pairs."""
    if x.shape[0] < 2:
        return float("nan")

    rng = np.random.default_rng(random_state)
    left = rng.integers(0, x.shape[0], size=n_pairs)
    right = rng.integers(0, x.shape[0], size=n_pairs)

    same_mask = left == right
    while np.any(same_mask):
        right[same_mask] = rng.integers(0, x.shape[0], size=int(np.sum(same_mask)))
        same_mask = left == right

    deltas = x[left] - x[right]
    distances = np.linalg.norm(deltas, axis=1)
    return float(np.mean(distances))


def _nearest_neighbor_distance(
    generated: np.ndarray,
    real: np.ndarray,
    batch_size: int,
) -> float:
    """Average nearest-neighbor distance from generated samples to real samples."""
    if generated.size == 0 or real.size == 0:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(real)

    distances = []
    for start in range(0, generated.shape[0], batch_size):
        batch = generated[start:start + batch_size]
        dist, _ = nn.kneighbors(batch, return_distance=True)
        distances.append(dist[:, 0])

    return float(np.mean(np.concatenate(distances, axis=0)))


def _plot_pca(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    save_path: str,
    random_state: int,
) -> None:
    """Plot a simple PCA projection comparing real and generated samples."""
    pca_input = np.vstack([real_samples, generated_samples])
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(pca_input)

    n_real = real_samples.shape[0]
    real_coords = coords[:n_real]
    gen_coords = coords[n_real:]

    plt.figure(figsize=(7, 5))
    plt.scatter(real_coords[:, 0], real_coords[:, 1], s=12, alpha=0.55, label="Real")
    plt.scatter(gen_coords[:, 0], gen_coords[:, 1], s=12, alpha=0.55, label="Generated")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: Real vs Generated Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _generate_for_label(
    cgan_model: Any,
    target_label: Any,
    n_generate: int,
    batch_size: int,
) -> np.ndarray:
    """Reuse the repo CGAN API while batching to avoid memory spikes."""
    parts = []
    remaining = int(max(0, n_generate))
    while remaining > 0:
        current = min(batch_size, remaining)
        batch = cgan_model.generate(
            labels_int=np.array([int(target_label)], dtype=np.int64),
            per_label=current,
        )
        batch_np = to_numpy(batch).astype(np.float32, copy=False)
        if batch_np.ndim == 1:
            batch_np = batch_np.reshape(-1, 1)
        parts.append(batch_np)
        remaining -= current

    if not parts:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(parts).astype(np.float32, copy=False)


def run_cgan_diagnostics(
    cgan_model: Any,
    real_features: Any,
    real_labels: Any,
    target_label: Any,
    output_dir: str,
    n_generate: int = 2000,
    batch_size: int = 256,
    random_state: int = 42,
    device: Optional[Any] = None,
    n_pairs: int = 5000,
    max_plot_samples: int = 1000,
    cgan_finetune_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run focused diagnostics for one CGAN target class and save artifacts under output_dir/cgan_diagnostics.
    """
    del device  # diagnostics currently reuses the model's own device handling

    diag_dir = os.path.join(output_dir, "cgan_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    real_x = to_numpy(real_features).astype(np.float32, copy=False)
    real_y = to_numpy(real_labels).reshape(-1)
    if real_x.ndim == 1:
        real_x = real_x.reshape(-1, 1)

    target_mask = real_y == target_label
    if real_y.size == real_x.shape[0] and np.any(target_mask):
        target_real = real_x[target_mask]
    else:
        target_real = real_x

    if target_real.shape[0] < 2:
        message = (
            f"Skipping CGAN diagnostics for target_label={target_label}: "
            f"need at least 2 real samples, found {target_real.shape[0]}."
        )
        metrics = {
            "target_label": target_label,
            "status": "skipped",
            "reason": message,
            "n_real_used": int(target_real.shape[0]),
            "n_generated_used": 0,
            "cgan_finetune_seconds": cgan_finetune_seconds,
            "generation_seconds": None,
        }
        with open(os.path.join(diag_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    generation_start = time.perf_counter()
    generated_x = _generate_for_label(
        cgan_model=cgan_model,
        target_label=target_label,
        n_generate=n_generate,
        batch_size=batch_size,
    )
    generation_seconds = time.perf_counter() - generation_start

    if generated_x.shape[0] == 0:
        raise RuntimeError("CGAN diagnostics could not generate any samples for the target label.")

    n_used = min(target_real.shape[0], generated_x.shape[0], int(max(2, n_generate)))
    real_eval = sample_rows(target_real, n=n_used, random_state=random_state).astype(np.float32, copy=False)
    gen_eval = sample_rows(generated_x, n=n_used, random_state=random_state + 1).astype(np.float32, copy=False)

    real_avg_pairwise_distance = _random_pairwise_distance_mean(
        real_eval, n_pairs=n_pairs, random_state=random_state
    )
    generated_avg_pairwise_distance = _random_pairwise_distance_mean(
        gen_eval, n_pairs=n_pairs, random_state=random_state + 1
    )
    diversity_ratio = (
        float(generated_avg_pairwise_distance / real_avg_pairwise_distance)
        if np.isfinite(real_avg_pairwise_distance) and abs(real_avg_pairwise_distance) > 1e-12
        else float("nan")
    )

    real_mean = np.mean(real_eval, axis=0)
    gen_mean = np.mean(gen_eval, axis=0)
    real_std = np.std(real_eval, axis=0)
    gen_std = np.std(gen_eval, axis=0)

    mean_abs_diff = np.abs(real_mean - gen_mean)
    std_abs_diff = np.abs(real_std - gen_std)
    nn_distance = _nearest_neighbor_distance(gen_eval, real_eval, batch_size=batch_size)

    plot_real = sample_rows(real_eval, n=min(max_plot_samples, real_eval.shape[0]), random_state=random_state)
    plot_gen = sample_rows(gen_eval, n=min(max_plot_samples, gen_eval.shape[0]), random_state=random_state + 1)
    _plot_pca(
        real_samples=plot_real,
        generated_samples=plot_gen,
        save_path=os.path.join(diag_dir, "pca_real_vs_generated.png"),
        random_state=random_state,
    )

    feature_stats = pd.DataFrame(
        {
            "feature_index": np.arange(real_eval.shape[1], dtype=int),
            "real_mean": real_mean,
            "generated_mean": gen_mean,
            "mean_abs_diff": mean_abs_diff,
            "real_std": real_std,
            "generated_std": gen_std,
            "std_abs_diff": std_abs_diff,
        }
    )
    feature_stats.to_csv(os.path.join(diag_dir, "feature_stats.csv"), index=False)

    metrics: Dict[str, Any] = {
        "target_label": int(target_label) if isinstance(target_label, (np.integer, int)) else target_label,
        "status": "ok",
        "n_real_used": int(real_eval.shape[0]),
        "n_generated_used": int(gen_eval.shape[0]),
        "real_avg_pairwise_distance": real_avg_pairwise_distance,
        "generated_avg_pairwise_distance": generated_avg_pairwise_distance,
        "diversity_ratio": diversity_ratio,
        "feature_mean_abs_diff_avg": float(np.mean(mean_abs_diff)),
        "feature_std_abs_diff_avg": float(np.mean(std_abs_diff)),
        "generated_to_real_nn_distance_avg": nn_distance,
        "cgan_finetune_seconds": None if cgan_finetune_seconds is None else float(cgan_finetune_seconds),
        "generation_seconds": float(generation_seconds),
        "n_pair_samples": int(n_pairs),
        "max_plot_samples": int(min(max_plot_samples, plot_real.shape[0], plot_gen.shape[0])),
    }

    with open(os.path.join(diag_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pairwise_stats = {
        "target_label": metrics["target_label"],
        "real_avg_pairwise_distance": real_avg_pairwise_distance,
        "generated_avg_pairwise_distance": generated_avg_pairwise_distance,
        "diversity_ratio": diversity_ratio,
        "n_pair_samples": int(n_pairs),
    }
    with open(os.path.join(diag_dir, "pairwise_distance_stats.json"), "w", encoding="utf-8") as f:
        json.dump(pairwise_stats, f, indent=2)

    pd.DataFrame([metrics]).to_csv(os.path.join(diag_dir, "metrics_summary.csv"), index=False)
    return metrics
