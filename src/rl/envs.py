from __future__ import annotations

from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# sklearn is only used as a fallback if the model doesn't expose .evaluate()
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class _BaseNIDSEnv(gym.Env):
    """
    Base class for NIDS fine-tuning environments.

    Observation (state):
        - Flattened, globally-normalized confusion matrix (C x C) -> shape (C*C,)

    Action (MultiDiscrete):
        - For each class c in {0..C-1}, an integer k_c in [0, max_per_class]
          indicating how many samples to draw for fine-tuning that class at this step.

    Reward:
        - Default: Δ macro-F1 (current - previous). You can switch to Δ accuracy via flag.

    Termination:
        - No terminal states by default; episode is truncated after `max_steps`.

    Model contract:
        - The classifier must expose `.predict(X) -> np.ndarray[int]`.
        - It SHOULD support incremental updates via `.partial_fit(X, y)`.
          If not available, `.fit(X, y)` will be used.

    Data contract:
        - train_df and test_df must contain a `label` column and feature columns.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        n_labels: int,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        model: Any,
        max_per_class: int = 10,
        max_steps: int = 50,
        use_delta_accuracy: bool = False,
        reward_beta: float = 0.0,
        reward_lambda_mode: str = "uniform",
        label_names: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert "label" in train_df.columns and "label" in test_df.columns, (
            "train_df and test_df must include a 'label' column"
        )
        self.n_labels = int(n_labels)
        self.model = model
        self.max_steps = int(max_steps)
        self.use_delta_accuracy = bool(use_delta_accuracy)
        self.reward_beta = float(reward_beta)
        self.reward_lambda_mode = str(reward_lambda_mode)
        self.label_names = list(label_names) if label_names is not None else [str(i) for i in range(self.n_labels)]

        # Fix feature ordering once
        self.data_cols = [c for c in train_df.columns if c != "label"]
        self.train_df = train_df[["label"] + self.data_cols].copy()
        self.test_df = test_df[["label"] + self.data_cols].copy()

        # Cached arrays for fast evaluation
        self.x_test = self.test_df[self.data_cols].values.astype(np.float32)
        self.y_test = self.test_df["label"].values.astype(int)

        # Gym spaces
        # MultiDiscrete expects values in [0, n_i - 1], so use n_i = max_per_class + 1 for 0..max_per_class
        self.max_per_class = int(max_per_class)
        self.max_total_samples_per_step = int(self.n_labels * self.max_per_class)
        self.action_space = spaces.MultiDiscrete([self.max_per_class + 1] * self.n_labels)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_labels * self.n_labels,), dtype=np.float32
        )

        # Episode state
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._prev_metrics: Dict[str, float] = {"acc": 0.0, "f1": 0.0}
        self._state = np.zeros(self.n_labels * self.n_labels, dtype=np.float32)
        self._lambda_vector = self._build_lambda_vector()

    def _build_lambda_vector(self) -> np.ndarray:
        """
        Construct per-class reward weights from the current class ordering.

        Modes:
        - uniform: all classes weighted equally
        - attack_priority: non-benign classes use a 2:1 ratio over benign
        - benign_priority: benign uses a 2:1 ratio over non-benign classes
        """
        labels = [str(name) for name in self.label_names]
        benign_mask = np.array(
            [("benign" in name.lower()) or ("normal" in name.lower()) for name in labels],
            dtype=bool,
        )
        weights = np.ones(self.n_labels, dtype=np.float32)

        if self.reward_lambda_mode == "uniform":
            pass
        elif self.reward_lambda_mode == "attack_priority":
            weights[~benign_mask] = 2.0
        elif self.reward_lambda_mode == "benign_priority":
            weights[benign_mask] = 2.0
        else:
            raise ValueError(
                f"Unsupported reward_lambda_mode='{self.reward_lambda_mode}'. "
                "Expected one of: uniform, attack_priority, benign_priority."
            )
        return weights

    @staticmethod
    def _per_class_recall(cm: np.ndarray) -> np.ndarray:
        row_sum = cm.sum(axis=1)
        recalls = np.divide(
            np.diag(cm),
            row_sum,
            out=np.zeros(cm.shape[0], dtype=np.float32),
            where=row_sum > 0,
        )
        return recalls.astype(np.float32, copy=False)

    # ---- hooks to override in subclasses ---------------------------------------------------
    def _sample_data_for_class(self, class_id: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) samples for the specified class.
        Subclasses must implement this to draw from real data or a generator.
        """
        raise NotImplementedError

    def _build_step_info(
        self,
        *,
        action: np.ndarray,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        total_samples_selected: int,
        cgan_samples_selected: int = 0,
        replay_samples_selected: int = 0,
        reward_terms: Optional[Dict[str, float]] = None,
        per_class_sample_counts: Optional[Dict[int, int]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metric_name = "acc" if self.use_delta_accuracy else "f1"
        delta_metric = float(post_metrics.get(metric_name, 0.0) - pre_metrics.get(metric_name, 0.0))
        total = int(total_samples_selected)
        cgan = int(cgan_samples_selected)
        replay = int(replay_samples_selected)
        info = {
            "step": self._t + 1,
            "metrics": post_metrics,
            "action": action,
            "pre_update_metric": float(pre_metrics.get(metric_name, 0.0)),
            "post_update_metric": float(post_metrics.get(metric_name, 0.0)),
            "delta_metric": delta_metric,
            "metric_name": metric_name,
            "total_samples_selected": total,
            "cgan_samples_selected": cgan,
            "replay_samples_selected": replay,
            "cgan_fraction": float(cgan / total) if total > 0 else 0.0,
            "replay_fraction": float(replay / total) if total > 0 else 0.0,
            "per_class_sample_counts": per_class_sample_counts or {},
            "reward_beta": self.reward_beta,
            "reward_lambda_mode": self.reward_lambda_mode,
        }
        if reward_terms:
            info.update(
                {
                    "progress_reward": float(reward_terms.get("progress_reward", 0.0)),
                    "efficiency_penalty": float(reward_terms.get("efficiency_penalty", 0.0)),
                    "reward": float(reward_terms.get("reward", 0.0)),
                }
            )
        if extra:
            info.update(extra)
        return info

    # ---- gym API ---------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        # Baseline evaluation
        self._prev_metrics = self._evaluate_metrics(self.x_test, self.y_test)
        self._prev_confusion = self._confusion(self.x_test, self.y_test)
        self._state = self._confusion_normalized(self.x_test, self.y_test)
        return self._state.copy(), {}

    def step(self, action: np.ndarray | list | tuple):
        # Validate & coerce action
        action = np.asarray(action, dtype=int).reshape(-1)
        if action.shape[0] != self.n_labels:
            raise ValueError(f"Action length {action.shape[0]} != n_labels {self.n_labels}")
        if np.any(action < 0) or np.any(action > self.max_per_class):
            raise ValueError("Action out of bounds for one or more classes")

        # Build fine-tuning mini-batch across classes
        X_parts, y_parts = [], []
        per_class_counts: Dict[int, int] = {}
        for cls_id, k in enumerate(action):
            if k <= 0:
                continue
            Xc, yc = self._sample_data_for_class(cls_id, int(k))
            if Xc.size == 0:
                continue
            X_parts.append(Xc.astype(np.float32, copy=False))
            y_parts.append(yc.astype(int, copy=False))
            per_class_counts[int(cls_id)] = int(yc.shape[0])

        pre_metrics = dict(self._prev_metrics)
        if X_parts:
            X_ft = np.vstack(X_parts)
            y_ft = np.concatenate(y_parts)
            self._incremental_update(X_ft, y_ft)
            total_samples_selected = int(y_ft.shape[0])
        else:
            total_samples_selected = 0

        # Evaluate & compute reward
        metrics = self._evaluate_metrics(self.x_test, self.y_test)
        curr_confusion = self._confusion(self.x_test, self.y_test)
        reward_terms = self._compute_reward_terms(
            self._prev_metrics,
            metrics,
            self._prev_confusion,
            curr_confusion,
            total_samples_selected=total_samples_selected,
        )
        self._prev_metrics = metrics
        self._prev_confusion = curr_confusion

        # Next state from confusion matrix
        self._state = self._confusion_normalized(self.x_test, self.y_test)

        self._t += 1
        terminated = False
        truncated = self._t >= self.max_steps
        info = self._build_step_info(
            action=action,
            pre_metrics=pre_metrics,
            post_metrics=metrics,
            total_samples_selected=total_samples_selected,
            reward_terms=reward_terms,
            per_class_sample_counts=per_class_counts,
        )
        return self._state.copy(), float(reward_terms["reward"]), terminated, truncated, info

    # ---- utilities -------------------------------------------------------------------------
    def _incremental_update(self, X: np.ndarray, y: np.ndarray) -> None:
        # Prefer partial_fit if available (online/incremental). Fallback to fit.
        if hasattr(self.model, "partial_fit"):
            try:
                self.model.partial_fit(X, y)
            except TypeError:
                # Some implementations may have (X, y, classes=...) signature on first call; try without.
                self.model.partial_fit(X, y)
        else:
            self.model.fit(X, y)

    def _confusion(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=np.arange(self.n_labels))
        return cm.astype(np.float32)

    def _confusion_normalized(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        cm = self._confusion(X, y)
        s = cm.sum()
        if s > 0:
            cm = cm / s
        return cm.reshape(-1).astype(np.float32)

    def _evaluate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # Prefer the model's own evaluator (ensures reward agrees with training loop)
        if hasattr(self.model, "evaluate"):
            out = self.model.evaluate(X, y)
            # Ensure keys exist
            acc = float(out.get("acc", 0.0))
            f1 = float(out.get("f1", 0.0))
            if (acc == 0.0 and f1 == 0.0) and "loss" in out:
                # Fallback to sklearn if custom evaluate didn't provide acc/f1
                y_pred = self.model.predict(X)
                acc = float(accuracy_score(y, y_pred))
                f1 = float(f1_score(y, y_pred, average="macro"))
            return {"acc": acc, "f1": f1}

        # Fallback: compute here
        y_pred = self.model.predict(X)
        acc = float(accuracy_score(y, y_pred))
        f1 = float(f1_score(y, y_pred, average="macro"))
        return {"acc": acc, "f1": f1}

    def _compute_reward_terms(
        self,
        prev: Dict[str, float],
        curr: Dict[str, float],
        prev_confusion: np.ndarray,
        curr_confusion: np.ndarray,
        *,
        total_samples_selected: int,
    ) -> Dict[str, float]:
        if self.use_delta_accuracy:
            progress_reward = float(curr["acc"] - prev["acc"])
        else:
            prev_recall = self._per_class_recall(prev_confusion)
            curr_recall = self._per_class_recall(curr_confusion)
            lambda_vec = self._lambda_vector / max(1.0, float(np.sum(self._lambda_vector)))
            progress_reward = float(np.sum(lambda_vec * (curr_recall - prev_recall)))

        # Treat beta as a real per-sample efficiency cost instead of a tiny
        # normalized fraction penalty. The mild convex term discourages
        # repeatedly requesting near-maximum budgets while keeping the reward
        # decomposition simple and aligned with the paper's penalty factor.
        sample_fraction = float(total_samples_selected / max(1, self.max_total_samples_per_step))
        efficiency_penalty = float(self.reward_beta * total_samples_selected * (1.0 + sample_fraction))
        reward = float(progress_reward - efficiency_penalty)
        return {
            "progress_reward": progress_reward,
            "efficiency_penalty": efficiency_penalty,
            "reward": reward,
        }


class NIDSEnvGym(_BaseNIDSEnv):
    """Environment that draws fine-tuning samples from the REAL training pool.

    For each class c, the action supplies k_c real samples from `train_df` to fine-tune the model.
    """

    def __init__(
        self,
        *,
        n_labels: int,
        test: pd.DataFrame,
        train_full: pd.DataFrame,
        model: Any,
        max_action: int = 10,
        max_step_size: int = 50,
        use_delta_accuracy: bool = False,
        reward_beta: float = 0.0,
        reward_lambda_mode: str = "uniform",
        label_names: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_labels=n_labels,
            test_df=test,
            train_df=train_full,
            model=model,
            max_per_class=max_action,
            max_steps=max_step_size,
            use_delta_accuracy=use_delta_accuracy,
            reward_beta=reward_beta,
            reward_lambda_mode=reward_lambda_mode,
            label_names=label_names,
            seed=seed,
        )
        # Pre-split per-class indices for fast sampling
        self._class_indices: Dict[int, np.ndarray] = {
            c: np.where(self.train_df["label"].values.astype(int) == c)[0]
            for c in range(self.n_labels)
        }

    def _sample_data_for_class(self, class_id: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._class_indices.get(int(class_id), np.array([], dtype=int))
        if idx.size == 0:
            F = len(self.data_cols)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        choice = self._rng.choice(idx, size=min(k, idx.size), replace=idx.size < k)
        batch = self.train_df.iloc[choice]
        X = batch[self.data_cols].values.astype(np.float32, copy=False)
        y = batch["label"].values.astype(int)  # <-- fixed: no second `.values`
        return X, y


class NIDSGANEnvGym(_BaseNIDSEnv):
    """Environment that draws fine-tuning samples from a Conditional GAN.

    The CGAN object must implement:
        - generate(labels_int: np.ndarray, per_label: int) -> np.ndarray[float]
          returning features (no label column) with shape (len(labels_int)*per_label, n_features)
    """

    def __init__(
        self,
        *,
        n_labels: int,
        test: pd.DataFrame,
        train_full: pd.DataFrame,
        model: Any,
        cgan: Any,
        max_action: int = 10,
        max_step_size: int = 50,
        use_delta_accuracy: bool = False,
        reward_beta: float = 0.0,
        reward_lambda_mode: str = "uniform",
        label_names: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_labels=n_labels,
            test_df=test,
            train_df=train_full,
            model=model,
            max_per_class=max_action,
            max_steps=max_step_size,
            use_delta_accuracy=use_delta_accuracy,
            reward_beta=reward_beta,
            reward_lambda_mode=reward_lambda_mode,
            label_names=label_names,
            seed=seed,
        )
        self.G = cgan

    def _sample_data_for_class(self, class_id: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            F = len(self.data_cols)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        # CGAN.generate expects integer labels and a per_label count
        Xg = self.G.generate(labels_int=np.array([int(class_id)]), per_label=int(k))
        if Xg is None or Xg.size == 0:
            F = len(self.data_cols)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        yg = np.full((Xg.shape[0],), int(class_id), dtype=int)
        return Xg.astype(np.float32, copy=False), yg


class NIDSHybridEnvGym(_BaseNIDSEnv):
    """Environment that draws fine-tuning samples from BOTH a replay buffer and a Conditional GAN.

    Action decoding (per class): [n_gan, n_replay] -> MultiDiscrete with nvec = [max_gen+1, max_replay+1] * C
    """

    def __init__(
        self,
        *,
        n_labels: int,
        test: pd.DataFrame,
        train_full: pd.DataFrame,
        model: Any,
        cgan: Any,
        replay_buffer: Any,
        max_gen_per_class: int = 10,
        max_replay_per_class: int = 10,
        max_step_size: int = 50,
        use_delta_accuracy: bool = False,
        reward_beta: float = 0.0,
        reward_lambda_mode: str = "uniform",
        label_names: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        # initialize base with a placeholder max_per_class; we'll override action_space below
        super().__init__(
            n_labels=n_labels,
            test_df=test,
            train_df=train_full,
            model=model,
            max_per_class=1,
            max_steps=max_step_size,
            use_delta_accuracy=use_delta_accuracy,
            reward_beta=reward_beta,
            reward_lambda_mode=reward_lambda_mode,
            label_names=label_names,
            seed=seed,
        )
        self.G = cgan
        self.replay = replay_buffer
        self.max_gen = int(max_gen_per_class)
        self.max_rep = int(max_replay_per_class)
        self.max_total_samples_per_step = int(self.n_labels * (self.max_gen + self.max_rep))

        # Override action space to pairs per class
        self.action_space = spaces.MultiDiscrete([(self.max_gen + 1), (self.max_rep + 1)] * self.n_labels)

    def step(self, action: np.ndarray | list | tuple):
        action = np.asarray(action, dtype=int).reshape(-1)
        if action.shape[0] != 2 * self.n_labels:
            raise ValueError(f"Action length {action.shape[0]} != 2*C ({2*self.n_labels})")

        a = action.reshape(self.n_labels, 2)
        if (a[:, 0] < 0).any() or (a[:, 0] > self.max_gen).any() or (a[:, 1] < 0).any() or (a[:, 1] > self.max_rep).any():
            raise ValueError("Action out of bounds for GAN or replay components")

        X_parts, y_parts = [], []
        per_class_counts: Dict[int, int] = {}
        per_class_breakdown: Dict[int, Dict[str, int]] = {}
        total_cgan_samples = 0
        total_replay_samples = 0
        pre_metrics = dict(self._prev_metrics)
        for cls in range(self.n_labels):
            n_gan, n_rep = int(a[cls, 0]), int(a[cls, 1])

            # CGAN samples
            if n_gan > 0:
                Xg = self.G.generate(labels_int=np.array([cls]), per_label=n_gan)
                if Xg is not None and Xg.size:
                    yg = np.full((Xg.shape[0],), cls, dtype=int)
                    X_parts.append(Xg.astype(np.float32, copy=False))
                    y_parts.append(yg)
                    total_cgan_samples += int(Xg.shape[0])
                    per_class_counts[cls] = per_class_counts.get(cls, 0) + int(Xg.shape[0])
                    cls_entry = per_class_breakdown.setdefault(cls, {"cgan": 0, "replay": 0})
                    cls_entry["cgan"] += int(Xg.shape[0])

            # Replay samples
            if n_rep > 0:
                Xr, yr = self.replay.sample_per_class(cls, n_rep)
                if Xr is not None and Xr.size:
                    X_parts.append(Xr.astype(np.float32, copy=False))
                    y_parts.append(yr.astype(int, copy=False))
                    total_replay_samples += int(Xr.shape[0])
                    per_class_counts[cls] = per_class_counts.get(cls, 0) + int(Xr.shape[0])
                    cls_entry = per_class_breakdown.setdefault(cls, {"cgan": 0, "replay": 0})
                    cls_entry["replay"] += int(Xr.shape[0])

        if X_parts:
            X_ft = np.vstack(X_parts)
            y_ft = np.concatenate(y_parts)
            self._incremental_update(X_ft, y_ft)
            total_samples_selected = int(y_ft.shape[0])
        else:
            total_samples_selected = 0

        # Evaluate & compute reward
        metrics = self._evaluate_metrics(self.x_test, self.y_test)
        curr_confusion = self._confusion(self.x_test, self.y_test)
        reward_terms = self._compute_reward_terms(
            self._prev_metrics,
            metrics,
            self._prev_confusion,
            curr_confusion,
            total_samples_selected=total_samples_selected,
        )
        self._prev_metrics = metrics
        self._prev_confusion = curr_confusion
        self._state = self._confusion_normalized(self.x_test, self.y_test)

        self._t += 1
        terminated = False
        truncated = self._t >= self.max_steps
        info = self._build_step_info(
            action=action,
            pre_metrics=pre_metrics,
            post_metrics=metrics,
            total_samples_selected=total_samples_selected,
            cgan_samples_selected=total_cgan_samples,
            replay_samples_selected=total_replay_samples,
            reward_terms=reward_terms,
            per_class_sample_counts=per_class_counts,
            extra={
                "buffer_fill": getattr(self.replay, "fill_ratio", lambda: {})(),
                "per_class_source_counts": per_class_breakdown,
            },
        )
        return self._state.copy(), float(reward_terms["reward"]), terminated, truncated, info
