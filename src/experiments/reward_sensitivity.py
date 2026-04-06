from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _beta_tag(beta: float) -> str:
    return str(beta).replace(".", "p")


def _plot_f1_heatmap(results_df: pd.DataFrame, beta_values: List[float], lambda_modes: List[str], save_path: str) -> None:
    heat = np.full((len(lambda_modes), len(beta_values)), np.nan, dtype=np.float32)
    for i, mode in enumerate(lambda_modes):
        for j, beta in enumerate(beta_values):
            row = results_df[
                (results_df["reward_lambda_mode"] == mode) &
                (results_df["reward_beta"] == beta)
            ]
            if not row.empty:
                heat[i, j] = float(row.iloc[0]["final_macro_f1_mean"])

    plt.figure(figsize=(7.5, 4.5))
    im = plt.imshow(heat, aspect="auto")
    plt.xticks(np.arange(len(beta_values)), [str(beta) for beta in beta_values])
    plt.yticks(np.arange(len(lambda_modes)), lambda_modes)
    plt.xlabel("beta")
    plt.ylabel("lambda_mode")
    plt.title("Reward Sensitivity: Final Macro-F1")
    plt.colorbar(im, label="final_macro_f1")

    for i in range(len(lambda_modes)):
        for j in range(len(beta_values)):
            value = heat[i, j]
            if np.isfinite(value):
                plt.text(j, i, f"{value:.3f}", ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_samples_vs_beta(results_df: pd.DataFrame, beta_values: List[float], lambda_modes: List[str], save_path: str) -> None:
    plt.figure(figsize=(7.5, 4.5))
    for mode in lambda_modes:
        sub = results_df[results_df["reward_lambda_mode"] == mode].sort_values("reward_beta")
        plt.plot(sub["reward_beta"], sub["late_window_mean_total_samples_selected_per_round_mean"], marker="o", label=mode)
    plt.xlabel("beta")
    plt.ylabel("late_window_mean_total_samples_selected_per_round")
    plt.title("Reward Sensitivity: Late-Stage Samples vs Beta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_reward_sensitivity_experiment(
    base_args: Any,
    beta_values: List[float],
    lambda_modes: List[str],
    output_dir: str,
    runner_fn: Callable[..., Dict[str, Any]],
    seeds: List[int],
) -> Dict[str, Any]:
    exp_dir = os.path.join(output_dir, "reward_sensitivity")
    runs_dir = os.path.join(exp_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for mode in lambda_modes:
        for beta in beta_values:
            for seed in seeds:
                run_args = copy.deepcopy(base_args)
                run_args.run_reward_sensitivity = False
                run_args.run_cgan_diagnostics = False
                run_args.run_rl_diagnostics = True
                run_args.reward_beta = float(beta)
                run_args.reward_lambda_mode = mode
                run_args.seed = int(seed)

                subrun_dir = os.path.join(runs_dir, f"lambda_{mode}__beta_{_beta_tag(beta)}__seed_{seed}")
                result = runner_fn(run_args, run_dir_override=subrun_dir)

                row = {
                    "seed": int(seed),
                    "reward_beta": float(beta),
                    "reward_lambda_mode": mode,
                    "final_macro_f1": float(result["final_eval"].get("f1", 0.0)),
                    "mean_post_update_delta_f1": float(result.get("rl_diag_metrics", {}).get("mean_post_update_delta_metric", 0.0)),
                    "best_post_update_delta_f1": float(result.get("rl_diag_metrics", {}).get("best_post_update_delta_metric", 0.0)),
                    "mean_total_samples_selected_per_round": float(result.get("rl_diag_metrics", {}).get("mean_total_samples_selected_per_round", 0.0)),
                    "mean_cgan_samples_selected_per_round": float(result.get("rl_diag_metrics", {}).get("mean_cgan_samples_selected_per_round", 0.0)),
                    "mean_replay_samples_selected_per_round": float(result.get("rl_diag_metrics", {}).get("mean_replay_samples_selected_per_round", 0.0)),
                    "final_cgan_fraction": float(result.get("rl_diag_metrics", {}).get("final_cgan_fraction", 0.0)),
                    "ppo_train_seconds": float(result.get("rl_diag_metrics", {}).get("ppo_train_seconds", 0.0)),
                    "total_adaptation_seconds": float(result.get("rl_diag_metrics", {}).get("total_adaptation_seconds", 0.0)),
                    "autonomy_rate": float(result.get("rl_diag_metrics", {}).get("autonomy_rate", 0.0)),
                    "mean_progress_reward_per_round": float(result.get("rl_diag_metrics", {}).get("mean_progress_reward_per_round", 0.0)),
                    "mean_efficiency_penalty_per_round": float(result.get("rl_diag_metrics", {}).get("mean_efficiency_penalty_per_round", 0.0)),
                    "final_efficiency_penalty": float(result.get("rl_diag_metrics", {}).get("final_efficiency_penalty", 0.0)),
                    "late_window_mean_total_samples_selected_per_round": float(result.get("rl_diag_metrics", {}).get("late_window_mean_total_samples_selected_per_round", 0.0)),
                    "late_window_mean_cgan_samples_selected_per_round": float(result.get("rl_diag_metrics", {}).get("late_window_mean_cgan_samples_selected_per_round", 0.0)),
                    "late_window_mean_replay_samples_selected_per_round": float(result.get("rl_diag_metrics", {}).get("late_window_mean_replay_samples_selected_per_round", 0.0)),
                    "late_window_mean_post_update_delta_metric": float(result.get("rl_diag_metrics", {}).get("late_window_mean_post_update_delta_metric", 0.0)),
                    "final_target_recall": float(result.get("final_target_recall", 0.0)),
                    "final_benign_recall": float(result.get("final_benign_recall", 0.0)),
                    "final_mean_attack_recall": float(result.get("final_mean_attack_recall", 0.0)),
                    "lambda_vector": result.get("reward_lambda_vector", []),
                    "label_names": result.get("label_names", []),
                    "run_dir": result.get("run_dir", ""),
                }
                rows.append(row)

    per_run_df = pd.DataFrame(rows).sort_values(["reward_lambda_mode", "reward_beta", "seed"]).reset_index(drop=True)
    per_run_csv = os.path.join(exp_dir, "reward_sensitivity_runs.csv")
    per_run_df.to_csv(per_run_csv, index=False)

    metric_cols = [
        "final_macro_f1",
        "mean_post_update_delta_f1",
        "best_post_update_delta_f1",
        "mean_total_samples_selected_per_round",
        "mean_cgan_samples_selected_per_round",
        "mean_replay_samples_selected_per_round",
        "final_cgan_fraction",
        "ppo_train_seconds",
        "total_adaptation_seconds",
        "autonomy_rate",
        "mean_progress_reward_per_round",
        "mean_efficiency_penalty_per_round",
        "final_efficiency_penalty",
        "late_window_mean_total_samples_selected_per_round",
        "late_window_mean_cgan_samples_selected_per_round",
        "late_window_mean_replay_samples_selected_per_round",
        "late_window_mean_post_update_delta_metric",
        "final_target_recall",
        "final_benign_recall",
        "final_mean_attack_recall",
    ]
    agg_df = per_run_df.groupby(["reward_lambda_mode", "reward_beta"], as_index=False)[metric_cols].agg(["mean", "std"])
    agg_df.columns = [
        "_".join([part for part in col if part]).strip("_")
        for col in agg_df.columns.to_flat_index()
    ]
    first_meta = per_run_df.groupby(["reward_lambda_mode", "reward_beta"], as_index=False).first()[
        ["reward_lambda_mode", "reward_beta", "lambda_vector", "label_names"]
    ]
    results_df = agg_df.merge(first_meta, on=["reward_lambda_mode", "reward_beta"], how="left")
    results_df = results_df.sort_values(["reward_lambda_mode", "reward_beta"]).reset_index(drop=True)

    results_csv = os.path.join(exp_dir, "reward_sensitivity_results.csv")
    results_json = os.path.join(exp_dir, "reward_sensitivity_results.json")
    results_df.to_csv(results_csv, index=False)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)

    _plot_f1_heatmap(
        results_df=results_df,
        beta_values=beta_values,
        lambda_modes=lambda_modes,
        save_path=os.path.join(exp_dir, "reward_sensitivity_heatmap_f1.png"),
    )
    _plot_samples_vs_beta(
        results_df=results_df,
        beta_values=beta_values,
        lambda_modes=lambda_modes,
        save_path=os.path.join(exp_dir, "reward_sensitivity_samples_vs_beta.png"),
    )

    best_idx = results_df["final_macro_f1_mean"].idxmax()
    best_row = results_df.iloc[int(best_idx)].to_dict()

    high_perf_threshold = float(results_df["final_macro_f1_mean"].max() - 0.01)
    high_perf_df = results_df[results_df["final_macro_f1_mean"] >= high_perf_threshold]
    low_sample_row = high_perf_df.sort_values("mean_total_samples_selected_per_round_mean").iloc[0].to_dict()

    stability = float(results_df["final_macro_f1_mean"].max() - results_df["final_macro_f1_mean"].min())
    stability_label = "stable" if stability < 0.02 else "highly variable"

    summary = {
        "beta_values": beta_values,
        "lambda_modes": lambda_modes,
        "seeds": seeds,
        "best_setting": best_row,
        "lowest_sample_high_performance_setting": low_sample_row,
        "f1_range": stability,
        "stability_assessment": stability_label,
        "results_csv": results_csv,
        "results_json": results_json,
        "per_run_csv": per_run_csv,
    }

    print(
        "[reward_sensitivity] best by final_macro_f1: "
        f"mode={best_row['reward_lambda_mode']} beta={best_row['reward_beta']} "
        f"f1={best_row['final_macro_f1_mean']:.4f}"
    )
    print(
        "[reward_sensitivity] lowest-sample high-performing setting: "
        f"mode={low_sample_row['reward_lambda_mode']} beta={low_sample_row['reward_beta']} "
        f"samples={low_sample_row['mean_total_samples_selected_per_round_mean']:.2f}"
    )
    print(f"[reward_sensitivity] performance across grid is {stability_label} (F1 range={stability:.4f})")
    return summary
