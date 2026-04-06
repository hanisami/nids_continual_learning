from __future__ import annotations
from typing import Dict, Optional, List, Any
from .ppo_agent import PPOAgent
import time
import numpy as np


def train_with_ppo(
    env,
    agent: PPOAgent,
    total_steps: int = 50000,
    rollout_len: int = 2048,
    log_every: int = 1000,
    eval_every: int = 5000,
    rl_history: Optional[List[Dict[str, Any]]] = None,
    retrain_history: Optional[List[Dict[str, Any]]] = None,
    timing_info: Optional[Dict[str, Any]] = None,
    metric_name: str = "f1",
) -> Dict[str, list]:
    """
    Minimal PPO training loop for continual-learning NIDS environments.

    - env: Gym environment (Hybrid, GAN, or Real)
    - agent: PPOAgent with MultiDiscrete action head
    - total_steps: Total environment interaction steps
    - rollout_len: Number of steps per PPO update
    - log_every: Print progress every ~log_every steps
    - eval_every: Frequency for evaluation logging (kept for future extensions)
    """
    logs = {"step": [], "reward": [], "acc": [], "f1": []}
    obs, _ = env.reset()
    step, ep_return = 0, 0.0
    start_perf = time.perf_counter()
    recent_returns = []
    inference_times_ms = []
    retraining_round_seconds = []
    rl_history = rl_history if rl_history is not None else []
    retrain_history = retrain_history if retrain_history is not None else []
    timing_info = timing_info if timing_info is not None else {}

    while step < total_steps:
        rollout_start = step
        # Collect rollout
        for _ in range(rollout_len):
            infer_start = time.perf_counter()
            action = agent.select_action(obs)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            inference_times_ms.append(float(infer_ms))
            round_start = time.perf_counter()
            next_obs, reward, terminated, truncated, info = env.step(action)
            round_seconds = time.perf_counter() - round_start
            retraining_round_seconds.append(float(round_seconds))
            done = terminated or truncated
            agent.store_outcome(reward, done)
            ep_return += reward
            obs = next_obs
            step += 1

            selected_total = int(info.get("total_samples_selected", 0)) if isinstance(info, dict) else 0
            selected_cgan = int(info.get("cgan_samples_selected", 0)) if isinstance(info, dict) else 0
            selected_replay = int(info.get("replay_samples_selected", 0)) if isinstance(info, dict) else 0
            step_metric_name = info.get("metric_name", metric_name) if isinstance(info, dict) else metric_name
            rl_history.append({
                "round_id": int(step),
                "env_step": int(info.get("step", step)) if isinstance(info, dict) else int(step),
                "policy_inference_ms": float(infer_ms),
                "retraining_round_seconds": float(round_seconds),
                "reward": float(reward),
                "progress_reward": float(info.get("progress_reward", 0.0)) if isinstance(info, dict) else 0.0,
                "efficiency_penalty": float(info.get("efficiency_penalty", 0.0)) if isinstance(info, dict) else 0.0,
                "metric_name": step_metric_name,
                "pre_update_metric": float(info.get("pre_update_metric", 0.0)) if isinstance(info, dict) else 0.0,
                "post_update_metric": float(info.get("post_update_metric", 0.0)) if isinstance(info, dict) else 0.0,
                "delta_metric": float(info.get("delta_metric", 0.0)) if isinstance(info, dict) else 0.0,
                "total_samples_selected": selected_total,
                "cgan_samples_selected": selected_cgan,
                "replay_samples_selected": selected_replay,
                "action": np.asarray(action, dtype=int).tolist(),
                "per_class_sample_counts": info.get("per_class_sample_counts", {}) if isinstance(info, dict) else {},
                "per_class_source_counts": info.get("per_class_source_counts", {}) if isinstance(info, dict) else {},
            })
            retrain_history.append({
                "round_id": int(step),
                "total_samples_selected": selected_total,
                "cgan_samples_selected": selected_cgan,
                "replay_samples_selected": selected_replay,
                "cgan_fraction": float(info.get("cgan_fraction", 0.0)) if isinstance(info, dict) else 0.0,
                "replay_fraction": float(info.get("replay_fraction", 0.0)) if isinstance(info, dict) else 0.0,
                "progress_reward": float(info.get("progress_reward", 0.0)) if isinstance(info, dict) else 0.0,
                "efficiency_penalty": float(info.get("efficiency_penalty", 0.0)) if isinstance(info, dict) else 0.0,
                "pre_update_metric": float(info.get("pre_update_metric", 0.0)) if isinstance(info, dict) else 0.0,
                "post_update_metric": float(info.get("post_update_metric", 0.0)) if isinstance(info, dict) else 0.0,
                "delta_metric": float(info.get("delta_metric", 0.0)) if isinstance(info, dict) else 0.0,
                "metric_name": step_metric_name,
                "per_class_sample_counts": info.get("per_class_sample_counts", {}) if isinstance(info, dict) else {},
            })

            if done:
                logs["step"].append(step)
                logs["reward"].append(ep_return)
                if isinstance(info, dict) and "metrics" in info:
                    logs["acc"].append(info["metrics"].get("acc", np.nan))
                    logs["f1"].append(info["metrics"].get("f1", np.nan))
                recent_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

            if step >= total_steps:
                break

        # PPO update
        if len(agent.rewards) >= agent.cfg.minibatch_size:
            agent.update()

        # Logging heartbeat
        if step % log_every < rollout_len:
            try:
                m = info.get("metrics", {})
                avg_ret = np.mean(recent_returns[-10:]) if recent_returns else 0.0
                print(
                    f"[PPO] step={step:>6d} | "
                    f"avgR={avg_ret:>7.4f} | "
                    f"acc={m.get('acc', float('nan')):>6.3f} | "
                    f"f1={m.get('f1', float('nan')):>6.3f}"
                )
            except Exception:
                pass

    total_adaptation_seconds = time.perf_counter() - start_perf
    print(f"\n[Training completed in {total_adaptation_seconds/60:.2f} min]")

    timing_info["ppo_train_seconds"] = float(total_adaptation_seconds)
    timing_info["policy_inference_ms"] = inference_times_ms
    timing_info["retraining_round_seconds"] = retraining_round_seconds
    timing_info["total_adaptation_seconds"] = float(total_adaptation_seconds)

    # Convert lists to arrays for downstream plotting
    for k in logs:
        logs[k] = np.asarray(logs[k], dtype=np.float32)

    return logs
