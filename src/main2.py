from __future__ import annotations

import os
import json
import argparse
import time
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Models & RL
from models.mlp_classifier import MLPClassifierTorch, MLPConfig
from models.cgan import ConditionalGANTorch as ConditionalGAN
from models.replay_buffer import ClassReplayBuffer
from experiments.cgan_diagnostics import run_cgan_diagnostics
from experiments.rl_diagnostics import run_rl_diagnostics
from experiments.reward_sensitivity import run_reward_sensitivity_experiment
from rl.envs import NIDSHybridEnvGym
from rl.ppo_agent import PPOAgent, PPOConfig
from rl.ppo_runner import train_with_ppo


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def set_global_seed(seed: Optional[int] = 42):
    if seed is None:
        return
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_dirs(base: str = "outputs") -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, ts)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return run_dir, ckpt_dir


def ensure_run_dirs(run_dir: str) -> Tuple[str, str]:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return run_dir, ckpt_dir


def detect_pre_dir(root: str) -> str:
    pre = os.path.join(root, "preprocessed")
    if not os.path.isdir(pre):
        raise FileNotFoundError(f"Expected folder: {pre}")
    return pre


def detect_files(pre_dir: str) -> Tuple[str, str, bool]:
    pq_train = os.path.join(pre_dir, "train_df.parquet")
    pq_test = os.path.join(pre_dir, "test_df.parquet")
    csv_train = os.path.join(pre_dir, "train_df.csv")
    csv_test = os.path.join(pre_dir, "test_df.csv")

    if os.path.exists(pq_train) and os.path.exists(pq_test):
        return pq_train, pq_test, True
    if os.path.exists(csv_train) and os.path.exists(csv_test):
        return csv_train, csv_test, False
    raise FileNotFoundError(
        f"Could not find train/test Parquet or CSV under {pre_dir}. "
        f"Looked for train_df.(parquet|csv) and test_df.(parquet|csv)."
    )


def build_label_in_where(ids: List[int]) -> str:
    if not ids:
        raise ValueError("build_label_in_where requires at least one label id")
    joined = ",".join(str(int(i)) for i in ids)
    return f"label in [{joined}]"


def load_label_dict(pre_dir: str) -> Dict[str, int]:
    path = os.path.join(pre_dir, "label_dict.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing label_dict.json in {pre_dir}. Re-run preprocessing.")
    with open(path, "r", encoding="utf-8") as f:
        names = list(json.load(f).keys())
    return {name: i for i, name in enumerate(names)}


# --------- feature detection: keep NUMERIC columns only (exclude strings like 'Attack') ---------

def get_feature_cols_parquet(train_path: str) -> list[str]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.types as pat

    # Read a pa.Schema (not ParquetSchema)
    schema: pa.Schema = pq.read_schema(train_path)

    def _is_numeric(pa_type: pa.DataType) -> bool:
        return pat.is_integer(pa_type) or pat.is_floating(pa_type) or pat.is_decimal(pa_type)

    feats = []
    for field in schema:  # type: ignore[assignment]
        name = field.name
        if name == "label":
            continue
        if _is_numeric(field.type):
            feats.append(name)
    return feats



def get_feature_cols_csv(train_path: str) -> List[str]:
    # sniff a small sample to infer which columns are numeric
    sample = pd.read_csv(train_path, nrows=2000)
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    # ensure we exclude the dense 'label' id from features
    return [c for c in numeric_cols if c != "label"]


# --------------------- streaming samplers ---------------------

def stream_sample_parquet(
    path: str,
    where: str,
    feature_cols: List[str],
    total_cap: int,
) -> pd.DataFrame:
    """
    Load up to total_cap rows from Parquet using a WHERE-like filter on 'label'.
    where: "all" | "label==k" | "label!=k" | "label in [1,2,3]"
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    cols = ["label"] + feature_cols

    # Base filter
    filt = ds.field("label").is_valid()
    if where != "all":
        if where.startswith("label=="):
            k = int(where.split("==")[1])
            filt = filt & (ds.field("label") == k)
        elif where.startswith("label!="):
            k = int(where.split("!=")[1])
            filt = filt & (ds.field("label") != k)
        elif where.startswith("label in"):
            inside = where.split("in", 1)[1].strip().strip("[]()")
            ids = [int(x) for x in inside.split(",") if x.strip() != ""]
            filt = filt & ds.field("label").isin(ids)
        else:
            raise ValueError(f"Unsupported where condition for parquet sampler: {where}")

    scanner = dataset.scanner(columns=cols, filter=filt)

    collected, n = [], 0
    for rb in scanner.to_batches():
        dfb = rb.to_pandas()
        collected.append(dfb)
        n += len(dfb)
        if n >= total_cap:
            break

    if not collected:
        return pd.DataFrame(columns=cols)

    out = pd.concat(collected, axis=0, ignore_index=True)

    # SAFETY: drop any non-numeric columns that slipped in
    numeric = out.select_dtypes(include=[np.number]).columns.tolist()
    keep = ["label"] + [c for c in feature_cols if c in numeric]
    out = out[keep]

    # enforce dtypes
    feat_kept = [c for c in out.columns if c != "label"]
    out[feat_kept] = out[feat_kept].astype(np.float32)
    out["label"] = out["label"].astype(int)

    if len(out) > total_cap:
        out = out.iloc[:total_cap].reset_index(drop=True)
    return out


def stream_sample_csv(
    path: str,
    feature_cols: List[str],
    label_filter,
    total_cap: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Load up to total_cap rows from CSV using pandas chunks and a Python filter function on the label.
    label_filter: callable(int)->bool (keep row if True) or None
    """
    # We may not know all columns reliably; read chunks with auto cols then trim
    collected, n = [], 0

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=True):
        # Keep only label + candidate numeric features we expect
        has = [c for c in chunk.columns if c == "label" or c in feature_cols]
        chunk = chunk[has]
        if "label" not in chunk.columns:
            continue

        # Coerce label to int (dropping non-numeric)
        chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce")
        chunk = chunk.dropna(subset=["label"])
        chunk["label"] = chunk["label"].astype(int)

        if label_filter is not None:
            chunk = chunk[chunk["label"].apply(label_filter)]
        if chunk.empty:
            continue

        # Drop non-numeric features just in case
        numeric = chunk.select_dtypes(include=[np.number]).columns.tolist()
        keep = ["label"] + [c for c in feature_cols if c in numeric]
        chunk = chunk[keep]

        feat_kept = [c for c in chunk.columns if c != "label"]
        if feat_kept:
            chunk[feat_kept] = chunk[feat_kept].astype(np.float32)

        collected.append(chunk)
        n += len(chunk)
        if n >= total_cap:
            break

    if not collected:
        return pd.DataFrame(columns=["label"] + feature_cols)

    out = pd.concat(collected, axis=0, ignore_index=True)
    if len(out) > total_cap:
        out = out.iloc[:total_cap].reset_index(drop=True)
    return out


def stratify_by_class(df: pd.DataFrame, per_class_cap: int, feature_cols: List[str]) -> pd.DataFrame:
    """Downsample to at most per_class_cap rows per class."""
    parts = []
    for cid, g in df.groupby("label"):
        if len(g) > per_class_cap:
            g = g.sample(n=per_class_cap, random_state=42)
        parts.append(g)
    out = pd.concat(parts, axis=0, ignore_index=True)
    if feature_cols:
        out[feature_cols] = out[feature_cols].astype(np.float32)
    out["label"] = out["label"].astype(int)
    return out


def plot_curves(run_dir: str, logs: Dict[str, np.ndarray]):
    if len(logs.get("step", [])) == 0:
        return
    # Reward
    plt.figure()
    plt.plot(logs["step"], logs["reward"], label="episodic return")
    plt.xlabel("env step")
    plt.ylabel("return")
    plt.title("PPO: episodic return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curve_return.png"), dpi=150)
    plt.close()

    # Acc / F1
    plt.figure()
    if len(logs.get("acc", [])) > 0:
        plt.plot(logs["step"][: len(logs["acc"])], logs["acc"], label="accuracy")
    if len(logs.get("f1", [])) > 0:
        plt.plot(logs["step"][: len(logs["f1"])], logs["f1"], label="macro-F1")
    plt.xlabel("env step (episode end)")
    plt.ylabel("score")
    plt.title("Evaluation on held-out test set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curve_metrics.png"), dpi=150)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Continual Learning (Hybrid: CGAN + Replay) with streaming data sampling")

    # Data / experiment
    ap.add_argument("--data_root", type=str, default=".", help="Project root that contains 'preprocessed/'")
    ap.add_argument("--target_class", type=str, required=True,
                    help="Class name to treat as UNSEEN for MLP pretrain (GAN trains on it)")
    ap.add_argument("--selected_classes", type=str, nargs="+", default=None,
                    help="Optional subset of classes to keep for the run; must include --target_class")
    # Sampling budgets (keep these modest to avoid OOM)
    ap.add_argument("--seen_cap_total", type=int, default=400_000,
                    help="Max rows to load for seen-class pretrain pool (total across all seen classes)")
    ap.add_argument("--seen_cap_per_class", type=int, default=50_000,
                    help="Per-class cap inside the seen pool; applied after initial load")
    ap.add_argument("--unseen_cap", type=int, default=120_000,
                    help="Rows of the target unseen class to load for CGAN training")
    ap.add_argument("--test_cap_per_class", type=int, default=20_000,
                    help="Rows to keep per class in test set for eval")

    # MLP
    ap.add_argument("--mlp_hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--mlp_dropout", type=float, default=0.1)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_weight_decay", type=float, default=0.0)
    ap.add_argument("--mlp_epochs", type=int, default=3)
    ap.add_argument("--mlp_batch", type=int, default=512)

    # CGAN
    ap.add_argument("--gan_noise", type=int, default=32)
    ap.add_argument("--gan_hidden", type=int, default=128)
    ap.add_argument("--gan_glayers", type=int, default=3)
    ap.add_argument("--gan_dlayers", type=int, default=3)
    ap.add_argument("--gan_lr", type=float, default=1e-3)
    ap.add_argument("--gan_batch", type=int, default=128)
    ap.add_argument("--gan_epochs", type=int, default=8)
    ap.add_argument("--run_cgan_diagnostics", action="store_true",
                    help="Run post-training CGAN diagnostics for the target class")
    ap.add_argument("--cgan_diag_samples", type=int, default=2000,
                    help="Number of generated samples to use for CGAN diagnostics")
    ap.add_argument("--cgan_diag_pairs", type=int, default=5000,
                    help="Number of random pairs used to estimate pairwise distances")
    ap.add_argument("--cgan_diag_max_plot_samples", type=int, default=1000,
                    help="Maximum real/generated samples to include in the PCA plot")
    ap.add_argument("--run_rl_diagnostics", action="store_true",
                    help="Run post-training RL diagnostics for autonomy, cost, and sample selection")
    ap.add_argument("--rl_diag_metric", type=str, default="f1",
                    help="Metric name to emphasize in RL diagnostics (default: f1)")
    ap.add_argument("--rl_diag_plot", action=argparse.BooleanOptionalAction, default=True,
                    help="Save the RL sample-selection plot")
    ap.add_argument("--run_reward_sensitivity", action="store_true",
                    help="Run a small reward-sensitivity sweep over lambda modes and beta values")
    ap.add_argument("--reward_beta", type=float, default=0.0,
                    help="Penalty factor beta applied as a per-sample efficiency cost in the reward")
    ap.add_argument("--reward_lambda_mode", type=str, default="uniform",
                    choices=["uniform", "attack_priority", "benign_priority"],
                    help="Per-class reward weighting scheme")
    ap.add_argument("--reward_sensitivity_seeds", type=int, nargs="+", default=[42, 7, 123],
                    help="Seeds to aggregate over for the reward-sensitivity experiment")

    # Replay buffer
    ap.add_argument("--replay_cap", type=int, default=20000, help="capacity per class")
    ap.add_argument("--seed_replay_per_class", type=int, default=2000,
                    help="initial real samples per seen class to seed buffer")

    # Env (hybrid: GAN + Replay)
    ap.add_argument("--max_gen", type=int, default=5, help="per-class max GAN samples per step")
    ap.add_argument("--max_rep", type=int, default=5, help="per-class max replay samples per step")
    ap.add_argument("--horizon", type=int, default=200, help="episode length (max steps)")
    ap.add_argument("--delta_acc", action="store_true", help="use Δaccuracy instead of ΔF1 as reward")

    # PPO
    ap.add_argument("--ppo_steps", type=int, default=50_000)
    ap.add_argument("--rollout_len", type=int, default=2048)
    ap.add_argument("--ppo_epochs", type=int, default=5)
    ap.add_argument("--ppo_batch", type=int, default=512)
    ap.add_argument("--ppo_lr_actor", type=float, default=3e-4)
    ap.add_argument("--ppo_lr_critic", type=float, default=1e-3)
    ap.add_argument("--ppo_gamma", type=float, default=0.99)
    ap.add_argument("--ppo_lam", type=float, default=0.95)
    ap.add_argument("--ppo_clip", type=float, default=0.2)
    ap.add_argument("--ppo_vf_coef", type=float, default=0.5)
    ap.add_argument("--ppo_ent_coef", type=float, default=0.01)
    ap.add_argument("--ppo_grad_clip", type=float, default=0.5)

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    return ap


def run_pipeline(args, run_dir_override: Optional[str] = None) -> Dict[str, object]:
    set_global_seed(args.seed)

    # Detect files & labels
    pre_dir = detect_pre_dir(args.data_root)
    train_path, test_path, is_parquet = detect_files(pre_dir)
    full_name_to_id = load_label_dict(pre_dir)
    if args.target_class not in full_name_to_id:
        raise SystemExit(f"[error] target_class='{args.target_class}' not found. Available (sample): "
                         f"{list(full_name_to_id.keys())[:10]} ...")
    if args.selected_classes:
        missing = [name for name in args.selected_classes if name not in full_name_to_id]
        if missing:
            raise SystemExit(f"[error] selected_classes contain unknown labels: {missing}")
        if args.target_class not in args.selected_classes:
            raise SystemExit("[error] --selected_classes must include --target_class")
        active_class_names = list(args.selected_classes)
    else:
        active_class_names = list(full_name_to_id.keys())

    active_full_ids = [int(full_name_to_id[name]) for name in active_class_names]
    seen_active_names = [name for name in active_class_names if name != args.target_class]
    seen_active_full_ids = [int(full_name_to_id[name]) for name in seen_active_names]
    target_full_id = int(full_name_to_id[args.target_class])
    name_to_id = {name: i for i, name in enumerate(active_class_names)}
    full_id_to_local_id = {int(full_name_to_id[name]): int(name_to_id[name]) for name in active_class_names}
    target_id = int(name_to_id[args.target_class])
    num_classes = len(name_to_id)

    # Feature columns (numeric only)
    feature_cols = get_feature_cols_parquet(train_path) if is_parquet else get_feature_cols_csv(train_path)

    # Build run dirs
    if run_dir_override is None:
        run_dir, ckpt_dir = build_run_dirs("outputs")
    else:
        run_dir, ckpt_dir = ensure_run_dirs(run_dir_override)
    with open(os.path.join(run_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "name_to_id": name_to_id,
                "target": args.target_class,
                "target_id": target_id,
                "selected_classes": active_class_names,
            },
            f,
            indent=2,
        )

    # ----------------------- Stream seen pool -----------------------
    print("[main] streaming seen-class pool (for MLP pretrain & replay seeding) ...")
    if is_parquet:
        seen_df = stream_sample_parquet(
            path=train_path,
            where=build_label_in_where(seen_active_full_ids),
            feature_cols=feature_cols,
            total_cap=args.seen_cap_total,
        )
    else:
        seen_df = stream_sample_csv(
            path=train_path,
            feature_cols=feature_cols,
            label_filter=lambda z: int(z) in set(seen_active_full_ids),
            total_cap=args.seen_cap_total,
        )
    if seen_df.empty:
        raise SystemExit("[error] seen_df is empty; increase caps or check preprocessing outputs.")

    # Optional per-class cap
    if args.seen_cap_per_class > 0:
        seen_df = stratify_by_class(seen_df, per_class_cap=args.seen_cap_per_class, feature_cols=feature_cols)

    # ----------------------- Stream unseen pool -----------------------
    print("[main] streaming UNSEEN target-class pool (for CGAN training) ...")
    if is_parquet:
        unseen_df = stream_sample_parquet(
            path=train_path,
            where=f"label=={target_full_id}",
            feature_cols=feature_cols,
            total_cap=args.unseen_cap,
        )
    else:
        unseen_df = stream_sample_csv(
            path=train_path,
            feature_cols=feature_cols,
            label_filter=lambda z: int(z) == target_full_id,
            total_cap=args.unseen_cap,
        )
    if unseen_df.empty:
        raise SystemExit("[error] unseen_df is empty; choose a different --target_class or increase --unseen_cap.")

    # ----------------------- Stream test (stratified) -----------------------
    print("[main] streaming test split (stratified per class) ...")
    if is_parquet:
        test_df = stream_sample_parquet(
            path=test_path, where=build_label_in_where(active_full_ids), feature_cols=feature_cols,
            total_cap=args.test_cap_per_class * num_classes
        )
    else:
        test_df = stream_sample_csv(
            path=test_path, feature_cols=feature_cols, label_filter=lambda z: int(z) in set(active_full_ids),
            total_cap=args.test_cap_per_class * num_classes
        )
    test_df = stratify_by_class(test_df, per_class_cap=args.test_cap_per_class, feature_cols=feature_cols)

    # Remap filtered labels to a dense local index for the active class subset.
    seen_df["label"] = seen_df["label"].map(full_id_to_local_id).astype(int)
    unseen_df["label"] = unseen_df["label"].map(full_id_to_local_id).astype(int)
    test_df["label"] = test_df["label"].map(full_id_to_local_id).astype(int)

    # Sanity prints
    print(f"[main] seen_df: {seen_df.shape} | unseen_df: {unseen_df.shape} | test_df: {test_df.shape}")

    # ----------------------- Prepare arrays -----------------------
    X_seen = seen_df[feature_cols].values.astype(np.float32)
    y_seen = seen_df["label"].values.astype(int)

    X_unseen = unseen_df[feature_cols].values.astype(np.float32)
    y_unseen = unseen_df["label"].values.astype(int)  # all equal to target_id

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(int)

    # ----------------------- MLP pretrain (seen classes only) -----------------------
    clf = MLPClassifierTorch(MLPConfig(
        input_dim=len(feature_cols),
        num_classes=num_classes,
        hidden_sizes=tuple(args.mlp_hidden),
        dropout=args.mlp_dropout,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        seed=args.seed,
        grad_clip=1.0,
        class_weight=None
    ))
    print("[main] pretraining MLP on seen classes only ...")
    clf.fit(X_seen, y_seen, epochs=args.mlp_epochs, batch_size=args.mlp_batch, verbose=1)
    pre_eval = clf.evaluate(X_test, y_test)
    print(f"[main] Pretrain eval: loss={pre_eval['loss']:.4f} acc={pre_eval['acc']:.4f} f1={pre_eval['f1']:.4f}")
    clf.save(os.path.join(ckpt_dir, "mlp_pretrain.pt"))

    # ----------------------- CGAN train (on unseen class only) -----------------------
    print("[main] training CGAN on the UNSEEN class ...")
    gan = ConditionalGAN(
        x_dim=len(feature_cols),
        num_classes=num_classes,
        noise_dim=args.gan_noise,
        min_units=args.gan_hidden,
        g_layers=args.gan_glayers,
        d_layers=args.gan_dlayers,
        batch_size=args.gan_batch,
        lr=args.gan_lr,
        seed=args.seed,
    )
    gan_train_start = time.perf_counter()
    gan.train(X_unseen, y_unseen, epochs=args.gan_epochs, verbose=True)
    cgan_finetune_seconds = time.perf_counter() - gan_train_start

    if args.run_cgan_diagnostics:
        print("[main] running CGAN diagnostics for the target class ...")
        diag_metrics = run_cgan_diagnostics(
            cgan_model=gan,
            real_features=X_unseen,
            real_labels=y_unseen,
            target_label=target_id,
            output_dir=run_dir,
            n_generate=args.cgan_diag_samples,
            batch_size=args.gan_batch,
            random_state=args.seed,
            device=getattr(gan, "device", None),
            n_pairs=args.cgan_diag_pairs,
            max_plot_samples=args.cgan_diag_max_plot_samples,
            cgan_finetune_seconds=cgan_finetune_seconds,
        )
        status = diag_metrics.get("status", "ok")
        print(f"[main] CGAN diagnostics status: {status}")
        if status == "skipped":
            print(f"[main] CGAN diagnostics reason: {diag_metrics.get('reason', 'n/a')}")

    # ----------------------- Replay buffer (seed with seen only) -----------------------
    print("[main] seeding replay buffer with seen classes ...")
    buf = ClassReplayBuffer(n_classes=num_classes, capacity_per_class=args.replay_cap,
                            feature_dim=len(feature_cols), seed=args.seed)
    if args.seed_replay_per_class > 0:
        per = int(args.seed_replay_per_class)
        for c, group in seen_df.groupby("label"):
            take = min(per, len(group))
            if take <= 0:
                continue
            sample = group.sample(n=take, random_state=args.seed)
            buf.add(sample[feature_cols].values.astype(np.float32),
                    sample["label"].values.astype(int))

    # ----------------------- Hybrid Environment -----------------------
    print("[main] building hybrid environment (GAN + Replay) ...")
    # Provide a minimal train_full (schema only) — hybrid doesn't sample real data directly
    train_schema = seen_df[["label"] + feature_cols].iloc[:1].copy()
    env = NIDSHybridEnvGym(
        n_labels=num_classes,
        test=test_df[["label"] + feature_cols],
        train_full=train_schema,      # schema placeholder
        model=clf,
        cgan=gan,
        replay_buffer=buf,
        max_gen_per_class=args.max_gen,
        max_replay_per_class=args.max_rep,
        max_step_size=args.horizon,
        use_delta_accuracy=args.delta_acc,
        reward_beta=args.reward_beta,
        reward_lambda_mode=args.reward_lambda_mode,
        label_names=list(name_to_id.keys()),
        seed=args.seed,
    )

    # ----------------------- PPO Agent & Training -----------------------
    cfg = PPOConfig(
        state_dim=env.observation_space.shape[0],
        nvec=env.action_space.nvec,             # [max_gen+1, max_rep+1] repeated C times
        lr_actor=args.ppo_lr_actor,
        lr_critic=args.ppo_lr_critic,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_lam,
        clip_eps=args.ppo_clip,
        vf_coef=args.ppo_vf_coef,
        ent_coef=args.ppo_ent_coef,
        max_grad_norm=args.ppo_grad_clip,
        epochs=args.ppo_epochs,
        minibatch_size=args.ppo_batch,
        seed=args.seed,
    )
    agent = PPOAgent(cfg)

    print("[main] training PPO ...")
    rl_history = [] if args.run_rl_diagnostics else None
    retrain_history = [] if args.run_rl_diagnostics else None
    rl_timing_info = {} if args.run_rl_diagnostics else None
    logs = train_with_ppo(
        env,
        agent,
        total_steps=args.ppo_steps,
        rollout_len=args.rollout_len,
        rl_history=rl_history,
        retrain_history=retrain_history,
        timing_info=rl_timing_info,
        metric_name=args.rl_diag_metric,
    )

    # ----------------------- Save & Plot -----------------------
    print("[main] saving artifacts ...")
    agent.save(os.path.join(ckpt_dir, "ppo_agent.pt"))
    clf.save(os.path.join(ckpt_dir, "mlp_final.pt"))
    # logs may be numpy arrays
    with open(os.path.join(run_dir, "logs.json"), "w", encoding="utf-8") as jf:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in logs.items()}, jf, indent=2)

    final = clf.evaluate(X_test, y_test)
    print(f"[main] Final eval: loss={final['loss']:.4f} acc={final['acc']:.4f} f1={final['f1']:.4f}")
    y_pred_final = clf.predict(X_test)
    final_cm = confusion_matrix(y_test, y_pred_final, labels=np.arange(num_classes)).astype(np.float32)
    row_sums = final_cm.sum(axis=1)
    final_recalls = np.divide(
        np.diag(final_cm),
        row_sums,
        out=np.zeros(num_classes, dtype=np.float32),
        where=row_sums > 0,
    )
    benign_mask = np.array([("benign" in name.lower()) or ("normal" in name.lower()) for name in name_to_id.keys()], dtype=bool)
    attack_mask = ~benign_mask
    plot_curves(run_dir, logs)
    rl_diag_metrics = None
    if args.run_rl_diagnostics and rl_history is not None and retrain_history is not None and rl_timing_info is not None:
        print("[main] saving RL diagnostics ...")
        rl_diag_metrics = run_rl_diagnostics(
            output_dir=run_dir,
            rl_history=rl_history,
            retrain_history=retrain_history,
            timing_info=rl_timing_info,
            metric_name=args.rl_diag_metric,
            plot=args.rl_diag_plot,
        )
    print(f"[main] run saved to: {run_dir}")
    return {
        "run_dir": run_dir,
        "final_eval": final,
        "pre_eval": pre_eval,
        "rl_diag_metrics": rl_diag_metrics or {},
        "reward_beta": args.reward_beta,
        "reward_lambda_mode": args.reward_lambda_mode,
        "reward_lambda_vector": env._lambda_vector.tolist(),
        "label_names": list(name_to_id.keys()),
        "final_target_recall": float(final_recalls[target_id]),
        "final_benign_recall": float(np.mean(final_recalls[benign_mask])) if np.any(benign_mask) else 0.0,
        "final_mean_attack_recall": float(np.mean(final_recalls[attack_mask])) if np.any(attack_mask) else 0.0,
    }


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.run_reward_sensitivity:
        parent_run_dir, _ = build_run_dirs("outputs")
        beta_values = [0.0, 0.005, 0.02, 0.1]
        lambda_modes = ["uniform", "attack_priority", "benign_priority"]
        summary = run_reward_sensitivity_experiment(
            base_args=args,
            beta_values=beta_values,
            lambda_modes=lambda_modes,
            output_dir=parent_run_dir,
            runner_fn=run_pipeline,
            seeds=list(args.reward_sensitivity_seeds),
        )
        with open(os.path.join(parent_run_dir, "reward_sensitivity", "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[main] reward sensitivity saved to: {os.path.join(parent_run_dir, 'reward_sensitivity')}")
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
