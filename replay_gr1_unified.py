#!/usr/bin/env python3
"""
replay_gr1_unified.py — replay a dataset trajectory in the SAME env the
training data was collected in: gr1_unified/Lift_GR1ArmsAndWaistFourierHands_Env
(from robocasa-gr1-tabletop-tasks).

Why this exists
---------------
The `diag_init_pose*.py` scripts build a robosuite `Lift` env with a custom
`controller_config.json` and try to drive it through the composite
controller + FourierRightHand wrapper. That path doesn't work — the
FourierRightHand's action interface silently no-ops most of what we send,
so the fingers barely move.

The `gr1_unified` env from robocasa-gr1-tabletop-tasks talks to the Fourier
hand NATIVELY through its per-part action dict (`action.right_hand` etc).
This is the env that generated your dataset, so it's the env we should
deploy in.

Two modes
---------
--dataset-format 44d
    Reads the raw 44-D dataset (e.g. ball_red_large_sim) whose parquets
    have a single `action` column of shape (44,) sliced as:
        [0:7]   left_arm
        [7:13]  left_hand   ← per-finger targets, 6 values
        [13:19] left_leg
        [19:22] neck
        [22:29] right_arm
        [29:35] right_hand  ← per-finger targets, 6 values
        [35:41] right_leg
        [41:44] waist
    Feeds these straight into the env. Best sanity check — if the cube
    isn't picked up here, the env is broken (or version-mismatched with
    the collection code).

--dataset-format 18d
    Reads the converted 18-D dataset (e.g. red_cube_small_sim) with
    action = [upper(17), r_trig(1)]. Reconstructs the 6-D per-finger
    action.right_hand from r_trig using a canonical grasp shape and
    max_r derived from a --canonical-dataset (a 44-D dataset from the
    same collection setup):
        action.right_hand[i] = r_trig * max_r * canonical_shape[i]
    This is the deploy-realistic path — matches what the VLA does at
    inference (VLA emits r_trig, we expand to per-finger).

Usage
-----
    # Sanity: pure 44-D replay
    python replay_gr1_unified.py \\
        --env-name gr1_unified/Lift_GR1ArmsAndWaistFourierHands_Env \\
        --dataset-format 44d \\
        --dataset ../ball_red_large_sim \\
        --episode 0 --video-dir ./replay_videos

    # Deploy-realistic: 18-D + canonical shape from a 44-D reference
    python replay_gr1_unified.py \\
        --env-name gr1_unified/Lift_GR1ArmsAndWaistFourierHands_Env \\
        --dataset-format 18d \\
        --dataset ../red_cube_small_sim \\
        --canonical-dataset ../ball_red_large_sim \\
        --episode 0 --video-dir ./replay_videos

Requires
--------
    gymnasium
    gr00t.eval.simulation (from Isaac-GR00T project)
    robocasa-gr1-tabletop-tasks (installed and registers `gr1_unified/*` envs)
        Setup: bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ── PFL mapping (same as diag_init_pose) for reading 18-D "upper" ──────────
# planner[i] = logical[PFL[i]] where logical = [waist(3), L_arm(7), R_arm(7)]
PLANNER_FROM_LOGICAL = [
    0, 1, 2,
    3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16,
]


# ─────────────────────────────────────────────────────────────────────
# Dataset loading — 44-D and 18-D layouts
# ─────────────────────────────────────────────────────────────────────
def _load_action_column(dataset_path: Path, episode: int) -> np.ndarray:
    p = Path(dataset_path) / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    df = pd.read_parquet(p)
    A = np.stack([np.asarray(x, dtype=np.float32) for x in df["action"]])
    print(f"[replay] loaded {p.name}: action shape {A.shape}")
    return A


def load_44d_actions(dataset_path: Path, episode: int) -> Dict[str, np.ndarray]:
    """Returns per-part action arrays (T, D_part). The 44-D layout is
    documented in convert_sim_to_no_legs.py:69 (SIM_SLICES)."""
    A = _load_action_column(dataset_path, episode)
    if A.shape[1] != 44:
        raise ValueError(
            f"Expected 44-D action, got {A.shape[1]}-D. "
            "For 18-D use --dataset-format 18d."
        )
    return {
        "left_arm":   A[:, 0:7],
        "left_hand":  A[:, 7:13],
        # left_leg (13:19), neck (19:22) → unused, env expects arms/waist only
        "right_arm":  A[:, 22:29],
        "right_hand": A[:, 29:35],
        # right_leg (35:41) → unused
        "waist":      A[:, 41:44],
    }


def load_18d_actions(dataset_path: Path, episode: int) -> Dict[str, np.ndarray]:
    """Returns (T, D_part) but with `right_hand` and `left_hand` as
    placeholders (T, 6) filled with zeros — caller must fill them in
    via reconstruct_hand_from_r_trig using a canonical shape."""
    A = _load_action_column(dataset_path, episode)
    if A.shape[1] != 18:
        raise ValueError(
            f"Expected 18-D action, got {A.shape[1]}-D. "
            "For 44-D use --dataset-format 44d."
        )
    T = A.shape[0]
    upper17 = A[:, :17]                    # (T, 17)
    r_trig  = A[:, 17]                     # (T,)

    # Undo planner permutation → [waist(3), L_arm(7), R_arm(7)]
    logical = np.empty((T, 17), dtype=np.float32)
    for planner_idx in range(17):
        logical[:, PLANNER_FROM_LOGICAL[planner_idx]] = upper17[:, planner_idx]

    return {
        "left_arm":   logical[:, 3:10],
        "left_hand":  np.zeros((T, 6), dtype=np.float32),   # filled later
        "right_arm":  logical[:, 10:17],
        "right_hand": np.zeros((T, 6), dtype=np.float32),   # filled later
        "waist":      logical[:, 0:3],
        "r_trig":     r_trig,
    }


# ─────────────────────────────────────────────────────────────────────
# Canonical grasp shape / max_r — inverts r_trig back to 6-D hand
# ─────────────────────────────────────────────────────────────────────
def compute_canonical_shape(canonical_dataset: Path, episode: int,
                             side: str = "right",
                             top_frac: float = 0.15) -> Tuple[np.ndarray, float]:
    """From a 44-D reference dataset, compute:
      * `max_r` — the max mean(|hand|) observed for this side over the
                  entire episode. This is the same normalizer used in
                  convert_sim_to_no_legs.py:143-153.
      * `canonical_shape` (6-vec, unit mean of |·|) — the "grasp shape"
                  averaged over the top `top_frac` of frames by mean(|hand|).
    """
    parts = load_44d_actions(canonical_dataset, episode)
    hand  = parts[f"{side}_hand"]                       # (T, 6)
    mean_abs = np.mean(np.abs(hand), axis=1)            # (T,)
    max_r = float(mean_abs.max())

    n_top = max(1, int(top_frac * len(hand)))
    top_idx = np.argsort(-mean_abs)[:n_top]
    top_hand = hand[top_idx].mean(axis=0)               # (6,)

    # Normalize to unit mean of |·| so canonical * (r_trig * max_r) recovers
    # a 6-vec whose mean(|·|) is exactly r_trig * max_r.
    denom = float(np.mean(np.abs(top_hand)))
    if denom < 1e-6:
        raise ValueError(f"canonical shape degenerate for {side} side")
    canonical = (top_hand / denom).astype(np.float32)

    print(f"[replay] {side}: max_r={max_r:.3f}   canonical_shape="
          f"{np.round(canonical, 3).tolist()}   "
          f"(mean of top {n_top} frames of {len(hand)})")
    return canonical, max_r


def reconstruct_hand_from_r_trig(r_trig: np.ndarray, canonical: np.ndarray,
                                   max_r: float) -> np.ndarray:
    """r_trig (T,) → hand (T, 6). Inverse of the forward map
       r_trig = clip(mean(|hand|) / max_r, 0, 1)  in convert_sim_to_no_legs."""
    r = np.clip(r_trig, 0.0, 1.0).astype(np.float32)
    return (r[:, None] * max_r) * canonical[None, :]     # (T, 6)


# ─────────────────────────────────────────────────────────────────────
# Env construction — reuse gr00t.eval.simulation helpers
# ─────────────────────────────────────────────────────────────────────
def build_env(env_name: str, video_dir: str, max_episode_steps: int,
              n_action_steps: int):
    """Wrap gym.make(env_name) with VideoRecordingWrapper + MultiStepWrapper
    exactly like simulation_service.py does. This keeps the action-dict
    interface (T=n_action_steps chunks) consistent with training-time."""
    import gymnasium as gym
    from gr00t.eval.sim import (
        SimulationConfig, VideoConfig, MultiStepConfig, _create_single_env,
    )

    config = SimulationConfig(
        env_name=env_name,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
        ),
    )
    env_fn = partial(_create_single_env, config=config, idx=0)
    env = gym.vector.SyncVectorEnv([env_fn])
    return env


# ─────────────────────────────────────────────────────────────────────
# Replay loop
# ─────────────────────────────────────────────────────────────────────
def run_replay(env, actions_by_key: Dict[str, np.ndarray], n_action_steps: int):
    """Feed the recorded actions to env.step in chunks of n_action_steps.
    Each chunk's shape must be (n_envs=1, n_action_steps, D_part).
    Args:
        actions_by_key: {"left_arm": (T,7), "right_arm": (T,7),
                         "left_hand": (T,6), "right_hand": (T,6),
                         "waist": (T,3)}
    """
    T = actions_by_key["right_arm"].shape[0]
    print(f"[replay] {T} timesteps, chunk size {n_action_steps}")
    obs, _ = env.reset()

    for t0 in range(0, T, n_action_steps):
        t1 = min(t0 + n_action_steps, T)
        chunk = t1 - t0

        def _pack(key):
            arr = actions_by_key[key][t0:t1]
            if chunk < n_action_steps:
                # Right-pad by repeating the last frame so shape matches.
                pad = np.repeat(arr[-1:], n_action_steps - chunk, axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            return arr[np.newaxis]         # (1, n_action_steps, D)

        actions = {
            "action.left_arm":   _pack("left_arm"),
            "action.left_hand":  _pack("left_hand"),
            "action.right_arm":  _pack("right_arm"),
            "action.right_hand": _pack("right_hand"),
            "action.waist":      _pack("waist"),
        }
        obs, rew, term, trunc, info = env.step(actions)
        success = bool(info.get("success", [[False]])[0][0]) if "success" in info else None
        print(f"[replay] chunk [{t0:3d}, {t1:3d})  "
              f"reward={float(rew[0]):+.3f}  "
              f"term={bool(term[0])}  trunc={bool(trunc[0])}  "
              f"success={success}")
        if bool(term[0]) or bool(trunc[0]):
            break

    env.close()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-name", required=True,
                   help="Gym-registered env id, e.g. "
                        "gr1_unified/Lift_GR1ArmsAndWaistFourierHands_Env")
    p.add_argument("--dataset", required=True, help="Dataset root or parquet folder")
    p.add_argument("--dataset-format", choices=("44d", "18d"), default="44d")
    p.add_argument("--canonical-dataset", default=None,
                   help="18-D mode: 44-D dataset to derive max_r + canonical "
                        "grasp shape from. Ignored in 44-D mode.")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--n-action-steps",   type=int, default=16)
    p.add_argument("--max-episode-steps",type=int, default=500)
    p.add_argument("--video-dir",        default="./replay_videos")

    args = p.parse_args()

    if args.dataset_format == "44d":
        actions_by_key = load_44d_actions(Path(args.dataset), args.episode)
    else:
        parts = load_18d_actions(Path(args.dataset), args.episode)
        if not args.canonical_dataset:
            raise SystemExit(
                "18-D mode requires --canonical-dataset pointing at a 44-D "
                "dataset to derive max_r + canonical grasp shape from.")
        canon_r, max_r = compute_canonical_shape(
            Path(args.canonical_dataset), args.episode, side="right")
        canon_l, max_l = compute_canonical_shape(
            Path(args.canonical_dataset), args.episode, side="left")
        parts["right_hand"] = reconstruct_hand_from_r_trig(
            parts["r_trig"], canon_r, max_r)
        # For the no-legs schema the model doesn't emit a left trigger; keep
        # the left hand open (matches training-time convention).
        parts["left_hand"] = np.zeros_like(parts["right_hand"])
        actions_by_key = {k: v for k, v in parts.items() if k != "r_trig"}

    env = build_env(
        env_name=args.env_name,
        video_dir=args.video_dir,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
    )
    run_replay(env, actions_by_key, args.n_action_steps)


if __name__ == "__main__":
    main()
