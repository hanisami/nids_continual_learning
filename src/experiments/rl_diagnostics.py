from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_max(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.asarray(values, dtype=np.float64)))


def _late_window(values: List[float]) -> List[float]:
    if not values:
        return []
    window = max(10, len(values) // 4)
    return values[-window:]


def _plot_selection_over_rounds(df: pd.DataFrame, save_path: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["round_id"], df["total_samples_selected"], label="total_samples_selected")
    plt.plot(df["round_id"], df["cgan_samples_selected"], label="cgan_samples_selected")
    plt.plot(df["round_id"], df["replay_samples_selected"], label="replay_samples_selected")
    plt.xlabel("Retraining round")
    plt.ylabel("Samples selected")
    plt.title("Sample Selection Over Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_rl_diagnostics(
    output_dir: str,
    rl_history: List[Dict[str, Any]],
    retrain_history: List[Dict[str, Any]],
    timing_info: Dict[str, Any],
    metric_name: str = "f1",
    plot: bool = True,
) -> Dict[str, Any]:
    diag_dir = os.path.join(output_dir, "rl_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    action_df = pd.DataFrame(rl_history)
    retrain_df = pd.DataFrame(retrain_history)

    if not action_df.empty:
        action_df.to_csv(os.path.join(diag_dir, "action_trace.csv"), index=False)
    else:
        pd.DataFrame(columns=["round_id"]).to_csv(os.path.join(diag_dir, "action_trace.csv"), index=False)

    if not retrain_df.empty:
        retrain_df.to_csv(os.path.join(diag_dir, "retraining_summary.csv"), index=False)
    else:
        pd.DataFrame(columns=["round_id"]).to_csv(os.path.join(diag_dir, "retraining_summary.csv"), index=False)

    total_samples = retrain_df["total_samples_selected"].tolist() if "total_samples_selected" in retrain_df else []
    cgan_samples = retrain_df["cgan_samples_selected"].tolist() if "cgan_samples_selected" in retrain_df else []
    replay_samples = retrain_df["replay_samples_selected"].tolist() if "replay_samples_selected" in retrain_df else []
    delta_metric = retrain_df["delta_metric"].tolist() if "delta_metric" in retrain_df else []
    progress_rewards = retrain_df["progress_reward"].tolist() if "progress_reward" in retrain_df else []
    efficiency_penalties = retrain_df["efficiency_penalty"].tolist() if "efficiency_penalty" in retrain_df else []
    late_total_samples = _late_window(total_samples)
    late_cgan_samples = _late_window(cgan_samples)
    late_replay_samples = _late_window(replay_samples)
    late_delta_metric = _late_window(delta_metric)

    total_decisions = int(len(action_df))
    auto_decisions = total_decisions

    metrics: Dict[str, Any] = {
        "metric_name": metric_name,
        "ppo_train_seconds": float(timing_info.get("ppo_train_seconds", 0.0)),
        "mean_policy_inference_ms": _safe_mean(timing_info.get("policy_inference_ms", [])),
        "max_policy_inference_ms": _safe_max(timing_info.get("policy_inference_ms", [])),
        "total_adaptation_seconds": float(timing_info.get("total_adaptation_seconds", 0.0)),
        "mean_retraining_round_seconds": _safe_mean(timing_info.get("retraining_round_seconds", [])),
        "num_retraining_rounds": int(len(retrain_df)),
        "retraining_round_indices": retrain_df["round_id"].tolist() if "round_id" in retrain_df else [],
        "number_of_policy_decisions": total_decisions,
        "policy_decisions_with_no_manual_override": auto_decisions,
        "autonomy_rate": float(auto_decisions / total_decisions) if total_decisions > 0 else 0.0,
        "mean_total_samples_selected_per_round": _safe_mean(total_samples),
        "mean_cgan_samples_selected_per_round": _safe_mean(cgan_samples),
        "mean_replay_samples_selected_per_round": _safe_mean(replay_samples),
        "final_total_samples_selected": float(total_samples[-1]) if total_samples else 0.0,
        "final_cgan_fraction": float(retrain_df["cgan_fraction"].iloc[-1]) if not retrain_df.empty and "cgan_fraction" in retrain_df else 0.0,
        "mean_post_update_delta_metric": _safe_mean(delta_metric),
        "best_post_update_delta_metric": _safe_max(delta_metric),
        "std_total_samples_selected_per_round": float(np.std(np.asarray(total_samples, dtype=np.float64))) if total_samples else 0.0,
        "mean_progress_reward_per_round": _safe_mean(progress_rewards),
        "mean_efficiency_penalty_per_round": _safe_mean(efficiency_penalties),
        "final_efficiency_penalty": float(efficiency_penalties[-1]) if efficiency_penalties else 0.0,
        "late_window_mean_total_samples_selected_per_round": _safe_mean(late_total_samples),
        "late_window_mean_cgan_samples_selected_per_round": _safe_mean(late_cgan_samples),
        "late_window_mean_replay_samples_selected_per_round": _safe_mean(late_replay_samples),
        "late_window_mean_post_update_delta_metric": _safe_mean(late_delta_metric),
    }

    with open(os.path.join(diag_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if plot and not retrain_df.empty:
        _plot_selection_over_rounds(
            retrain_df,
            os.path.join(diag_dir, "sample_selection_over_rounds.png"),
        )

    return metrics
