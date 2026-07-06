#!/usr/bin/env python3
"""
convert_sim_to_no_legs.py — convert a GR1ArmsOnly absolute-next-state sim
dataset into the G1 "no-legs" schema used by deploy_groot.py and the real
teleop datasets.

Output is schema-compatible with `modality_config_gr1nyu_no_legs.py`:

    state  (28,)  =  left_arm(7) + left_hand(7) + right_arm(7) + right_hand(7)
    action (18,)  =  upper(17) + r_trig(1)

Conversion rules (no IK retargeting — joint values are copied as-is):

1. STATE
   * left_arm[0:7]   ← sim_state[0:7]   (sim left_arm, 7 dims)
   * right_arm[0:7]  ← sim_state[22:29] (sim right_arm, 7 dims)
   * left_hand[7]    ← analog grip ∈ [0,1] derived from sim_state[7:13]
                       (mean of absolute finger angles, normalized by dataset max)
     left_hand[8:14] ← zeros (the no-legs schema reserves 7 dims per hand
                       slot, but only one carries the meaningful signal)
   * right_hand same idea, derived from sim_state[29:35]

   sim_state's left_leg, neck, right_leg, waist are DROPPED — the no-legs
   schema doesn't have those keys.

2. ACTION (the sim's action is the next absolute state of all 44 joints;
   we keep only upper-body + derive right grip)
   * upper[17] is the 17-joint planner order (UPPER_BODY_INDICES_OLD), built
     from sim_action's waist(3) + left_arm(7) + right_arm(7), interleaved as:
        waist_yaw, waist_roll, waist_pitch,
        L_sh_pitch, R_sh_pitch, L_sh_roll, R_sh_roll,
        L_sh_yaw,  R_sh_yaw,  L_elbow,    R_elbow,
        L_wr_roll, R_wr_roll, L_wr_pitch, R_wr_pitch,
        L_wr_yaw,  R_wr_yaw
   * r_trig ← grip(sim_action[29:35])   (right-hand closure in the next frame)

3. VIDEO is copied verbatim — already 256×256 mp4 with key
   `observation.images.ego_view`, which matches the no-legs modality config.

4. METADATA is rewritten to match the no-legs teleop format:
   * info.json   — features rewritten with new state/action shapes
   * modality.json — copied from a reference no-legs dataset (4-key state,
                     2-key action, ego_view video, same annotation keys)
   * tasks.jsonl / episodes.jsonl — preserved unchanged (same task strings)

Usage:

    /home/mim-server/miniconda3/envs/lerobot_g1/bin/python \\
        gr00t_training_context/convert_sim_to_no_legs.py \\
        --src gr00t_training_context/ball_red_large_sim \\
        --dst gr00t_training_context/ball_red_large_sim_no_legs

A reference no-legs dataset is read to source the exact modality.json /
info.json layout the trainer expects.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ── Sim 44-dim modality slices (per ball_red_large_sim/meta/modality.json) ──
SIM_SLICES = {
    "left_arm":   slice(0, 7),
    "left_hand":  slice(7, 13),
    "left_leg":   slice(13, 19),
    "neck":       slice(19, 22),
    "right_arm":  slice(22, 29),
    "right_hand": slice(29, 35),
    "right_leg":  slice(35, 41),
    "waist":      slice(41, 44),
}

# ── No-legs 28-dim state slices (per modality_config_gr1nyu_no_legs.py) ──
NO_LEGS_STATE_SLICES = {
    "left_arm":   slice(0, 7),
    "left_hand":  slice(7, 14),
    "right_arm":  slice(14, 21),
    "right_hand": slice(21, 28),
}

# 17-joint planner-order action, indexing into a logical {waist(3), L_arm(7),
# R_arm(7)} concat. Source index = position in the concatenated 17-vector
# (waist 0..2, L_arm 3..9, R_arm 10..16); destination is planner order.
PLANNER_FROM_LOGICAL = [
    0,   # planner[0]  = waist_yaw    ← waist[0]
    1,   # planner[1]  = waist_roll   ← waist[1]
    2,   # planner[2]  = waist_pitch  ← waist[2]
    3,   # planner[3]  = L_sh_pitch   ← L_arm[0]
    10,  # planner[4]  = R_sh_pitch   ← R_arm[0]
    4,   # planner[5]  = L_sh_roll    ← L_arm[1]
    11,  # planner[6]  = R_sh_roll    ← R_arm[1]
    5,   # planner[7]  = L_sh_yaw     ← L_arm[2]
    12,  # planner[8]  = R_sh_yaw     ← R_arm[2]
    6,   # planner[9]  = L_elbow      ← L_arm[3]
    13,  # planner[10] = R_elbow      ← R_arm[3]
    7,   # planner[11] = L_wr_roll    ← L_arm[4]
    14,  # planner[12] = R_wr_roll    ← R_arm[4]
    8,   # planner[13] = L_wr_pitch   ← L_arm[5]
    15,  # planner[14] = R_wr_pitch   ← R_arm[5]
    9,   # planner[15] = L_wr_yaw     ← L_arm[6]
    16,  # planner[16] = R_wr_yaw     ← R_arm[6]
]


def grip_from_hand_state(hand_state_6dim: np.ndarray,
                         max_grip: float) -> float:
    """Map a 6-DOF Fourier dex3 finger state to a single analog [0,1] grip.

    Heuristic: mean of absolute finger-joint angles, normalized by the
    dataset-wide maximum mean(|hand|) value and clipped to [0,1]. Open hand
    (all zeros) → 0; fully closed → ~1.
    """
    s = float(np.mean(np.abs(hand_state_6dim)))
    if max_grip <= GRIP_DEAD_BAND:
        return 0.0
    return float(np.clip(s / max_grip, 0.0, 1.0))


# If the max observed mean(|hand|) is below this, the hand was effectively
# stationary across the whole dataset (just sensor noise around zero). In
# that case dividing by max_grip would amplify the noise into spurious
# grip-closure signal; treat the side as "always open" and output zeros.
GRIP_DEAD_BAND = 0.1   # radians


def grip_from_hand_state_batch(hand_state_T6: np.ndarray,
                               max_grip: float) -> np.ndarray:
    """Vectorized version returning shape (T,) float32 grip values."""
    if max_grip <= GRIP_DEAD_BAND:
        return np.zeros(hand_state_T6.shape[0], dtype=np.float32)
    return np.clip(
        np.mean(np.abs(hand_state_T6), axis=1) / max_grip, 0.0, 1.0
    ).astype(np.float32)


def compute_dataset_max_grip(parquet_paths: list[Path]) -> tuple[float, float]:
    """First pass — find max mean(|hand|) across the whole dataset (per side)."""
    max_l, max_r = 0.0, 0.0
    for p in parquet_paths:
        df = pd.read_parquet(p)
        s = np.stack([np.asarray(x, dtype=np.float32) for x in df['observation.state'].values])
        lh = s[:, SIM_SLICES['left_hand']]
        rh = s[:, SIM_SLICES['right_hand']]
        max_l = max(max_l, float(np.mean(np.abs(lh), axis=1).max()) if len(lh) else 0.0)
        max_r = max(max_r, float(np.mean(np.abs(rh), axis=1).max()) if len(rh) else 0.0)
    return max_l, max_r


def convert_state_array(state_T44: np.ndarray,
                        max_l: float, max_r: float,
                        empty_hand_proprio: bool = False) -> np.ndarray:
    """(T, 44) sim state → (T, 28) no-legs state.

    Hand slots are 7-dim each in the no-legs schema. Two modes:

    * Default (``empty_hand_proprio=False``): preserve the sim's full 6-DOF
      Fourier dex3 finger state PLUS an analog grip scalar:
          slot[0]   = analog grip ∈ [0,1]  (single scalar summary, easy to read)
          slot[1:7] = the six sim hand-state joints, copied verbatim
      The model sees both a summary signal and the per-joint finger angles.

    * ``empty_hand_proprio=True``: mimic the teleop datasets, which carry
      essentially no usable hand proprioception (the Unitree dex3 defaults
      are constant garbage). Just the grip scalar in slot[0]; slot[1:7] are
      zeros. Use this when you want the converted sim data to expose the
      model to the SAME (uninformative) hand-proprio statistics that real
      teleop data has — important if you're trying to make the model rely
      only on vision for grasp timing, matching teleop conditions.
    """
    T = state_T44.shape[0]
    out = np.zeros((T, 28), dtype=np.float64)
    out[:, 0:7]   = state_T44[:, SIM_SLICES['left_arm']]
    out[:, 14:21] = state_T44[:, SIM_SLICES['right_arm']]
    # Grip scalars always go in slot[0] of each hand block, regardless of mode.
    out[:, 7]  = grip_from_hand_state_batch(state_T44[:, SIM_SLICES['left_hand']],  max_l)
    out[:, 21] = grip_from_hand_state_batch(state_T44[:, SIM_SLICES['right_hand']], max_r)
    if not empty_hand_proprio:
        # Slots [8:14] and [22:28] each take the 6 sim hand-state dims.
        out[:, 8:14]  = state_T44[:, SIM_SLICES['left_hand']]
        out[:, 22:28] = state_T44[:, SIM_SLICES['right_hand']]
    # else: leave [8:14] / [22:28] at zero (empty hand proprio).
    return out


def convert_action_array(action_T44: np.ndarray, max_r: float) -> np.ndarray:
    """(T, 44) sim action → (T, 18) no-legs action.

    Sim's action is the next absolute state — we extract upper-body + derive
    the right-hand grip. Lower-body / neck / left-hand action components are
    dropped (the no-legs action space doesn't carry them).
    """
    T = action_T44.shape[0]
    # Build the logical 17-vector: [waist(3), left_arm(7), right_arm(7)]
    waist = action_T44[:, SIM_SLICES['waist']]      # (T, 3)
    larm  = action_T44[:, SIM_SLICES['left_arm']]    # (T, 7)
    rarm  = action_T44[:, SIM_SLICES['right_arm']]   # (T, 7)
    logical_17 = np.concatenate([waist, larm, rarm], axis=1)   # (T, 17)
    upper = logical_17[:, PLANNER_FROM_LOGICAL]      # (T, 17) in planner order

    rh_action = action_T44[:, SIM_SLICES['right_hand']]   # (T, 6)
    r_trig = grip_from_hand_state_batch(rh_action, max_r) # (T,)

    out = np.zeros((T, 18), dtype=np.float64)
    out[:, 0:17] = upper
    out[:, 17]   = r_trig
    return out


def convert_episode(in_path: Path, out_path: Path,
                    max_l: float, max_r: float,
                    empty_hand_proprio: bool = False) -> int:
    """Read one sim parquet, write the converted no-legs parquet. Returns T."""
    df = pd.read_parquet(in_path)
    state_T44 = np.stack([np.asarray(x, dtype=np.float32)
                          for x in df['observation.state'].values])
    action_T44 = np.stack([np.asarray(x, dtype=np.float32)
                           for x in df['action'].values])

    new_state  = convert_state_array(state_T44, max_l, max_r,
                                      empty_hand_proprio=empty_hand_proprio)
    new_action = convert_action_array(action_T44, max_r)

    # Build the new dataframe with the same auxiliary columns the teleop
    # datasets carry, so the LeRobot loader sees a familiar shape.
    out_df = pd.DataFrame({
        'observation.state': [row.tolist() for row in new_state],
        'action':            [row.tolist() for row in new_action],
        'timestamp': df['timestamp'].values if 'timestamp' in df.columns
                     else np.arange(len(df)) / 20.0,
        'annotation.human.action.task_description': df.get(
            'annotation.human.action.task_description',
            np.zeros(len(df), dtype=np.float32),
        ),
        'task_index': df.get('task_index', np.zeros(len(df), dtype=np.float32)),
        'annotation.human.validity': df.get(
            'annotation.human.validity',
            np.ones(len(df), dtype=np.float32),
        ),
        'episode_index': df['episode_index'] if 'episode_index' in df.columns
                         else np.zeros(len(df), dtype=np.int64),
        'index': df['index'] if 'index' in df.columns
                 else np.arange(len(df), dtype=np.int64),
        'next.reward': df.get('next.reward', np.zeros(len(df), dtype=np.float32)),
        'next.done': df.get('next.done', np.zeros(len(df), dtype=np.float32)),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    return len(df)


def write_meta(src_meta: Path, dst_meta: Path, *,
               num_episodes: int, num_frames: int, fps: float,
               max_l: float, max_r: float,
               empty_hand_proprio: bool = False) -> None:
    """Write info.json (with the no-legs feature shapes), modality.json (no-legs
    schema), and copy tasks.jsonl / episodes.jsonl from the source."""
    dst_meta.mkdir(parents=True, exist_ok=True)

    # ── modality.json: hard-coded no-legs schema ───────────────────
    modality = {
        "state": {
            "left_arm":   {"start": 0,  "end": 7},
            "left_hand":  {"start": 7,  "end": 14},
            "right_arm":  {"start": 14, "end": 21},
            "right_hand": {"start": 21, "end": 28},
        },
        "action": {
            "upper":  {"start": 0,  "end": 17},
            "r_trig": {"start": 17, "end": 18},
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        },
    }
    (dst_meta / "modality.json").write_text(json.dumps(modality, indent=4))

    # ── info.json: clone fields where reasonable, rewrite the schema-
    # dependent shapes for state(28) and action(18). ────────────────
    src_info = json.loads((src_meta / "info.json").read_text())
    info = dict(src_info)   # shallow copy
    info["total_episodes"] = int(num_episodes)
    info["total_frames"]   = int(num_frames)
    info["total_videos"]   = int(num_episodes)
    info["chunks_size"]    = int(num_episodes)
    info["total_chunks"]   = 1
    info["robot_type"]     = "GR1ArmsOnly_no_legs"  # marker so it's clear this is the converted variant
    info["fps"]            = float(fps)
    info["splits"]         = {"train": "0:100"}
    info["data_path"]      = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"]     = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    # Rebuild the features dict to match the teleop no-legs layout exactly.
    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": [28],
            "names": [f"motor_{i}" for i in range(28)],
        },
        "action": {
            "dtype": "float64",
            "shape": [18],
            "names": [f"motor_{i}" for i in range(18)],
        },
        "timestamp":  {"dtype": "float64", "shape": [1]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64",   "shape": [1]},
        "annotation.human.validity": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index":          {"dtype": "int64", "shape": [1]},
        "next.reward":    {"dtype": "float64", "shape": [1]},
        "next.done":      {"dtype": "bool",    "shape": [1]},
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
    }
    info["features"] = features
    # Stash conversion provenance for downstream debugging.
    if empty_hand_proprio:
        hand_layout_note = ("hand state slot[0] = analog grip ∈ [0,1]; "
                            "slot[1:7] = zeros (empty hand proprio, matches teleop)")
    else:
        hand_layout_note = ("hand state slot[0] = analog grip ∈ [0,1]; "
                            "slot[1:7] = the 6-DOF sim dex3 finger angles "
                            "(preserves full hand proprio)")
    info["conversion"] = {
        "source": "GR1ArmsOnly absolute next-state sim",
        "empty_hand_proprio": bool(empty_hand_proprio),
        "grip_normalization": {
            "max_mean_abs_left_hand":  max_l,
            "max_mean_abs_right_hand": max_r,
            "method": "grip = clip(mean(|hand_state|) / max_observed, 0, 1); "
                      "if max_observed <= 0.1 rad (dead band), grip is forced to 0",
        },
        "notes": ("joint values copied without IK retargeting; "
                  + hand_layout_note +
                  "; action upper(17) in planner order matching "
                  "UPPER_BODY_INDICES_OLD; r_trig derived from right-hand "
                  "next-state"),
    }
    (dst_meta / "info.json").write_text(json.dumps(info, indent=4))

    # ── pass-through metadata ──────────────────────────────────────
    for fname in ("tasks.jsonl", "episodes.jsonl"):
        src_f = src_meta / fname
        if src_f.exists():
            shutil.copy2(src_f, dst_meta / fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="Source sim dataset root (must contain meta/, data/, videos/).")
    p.add_argument("--dst", required=True,
                   help="Destination dataset root (created if missing).")
    p.add_argument("--copy-videos", action="store_true", default=True,
                   help="Copy the mp4s into the destination (default: yes).")
    p.add_argument("--symlink-videos", action="store_true",
                   help="Symlink instead of copying to save disk.")
    p.add_argument("--empty-hand-proprio", action="store_true",
                   help="Zero out hand-state proprioception (slots [8:14] and "
                        "[22:28] of the state vector), keeping only the analog "
                        "grip scalar in slot[7]/[21]. Use this to match the "
                        "teleop datasets' uninformative hand proprio. Default: "
                        "preserve the sim's 6-DOF dex3 finger state alongside "
                        "the grip scalar (more informative training signal).")
    args = p.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"src does not look like a LeRobot dataset: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # ── Parquets ───────────────────────────────────────────────────
    src_data_dir = src / "data" / "chunk-000"
    dst_data_dir = dst / "data" / "chunk-000"
    parquet_paths = sorted(src_data_dir.glob("episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquets in {src_data_dir}")

    print(f"[convert] First pass — measuring hand-state ranges across "
          f"{len(parquet_paths)} episodes...")
    max_l, max_r = compute_dataset_max_grip(parquet_paths)
    print(f"[convert]   max(mean|left_hand|)  = {max_l:.4f}")
    print(f"[convert]   max(mean|right_hand|) = {max_r:.4f}")
    if max_l == 0 and max_r == 0:
        print("[convert]   WARNING: both hand-state mean(|x|) maxes are zero — "
              "grip channel will be all-zero in the output.")

    print(f"[convert] Second pass — converting parquets...")
    total_frames = 0
    for in_path in parquet_paths:
        out_path = dst_data_dir / in_path.name
        T = convert_episode(in_path, out_path, max_l, max_r,
                            empty_hand_proprio=args.empty_hand_proprio)
        total_frames += T
    print(f"[convert]   wrote {len(parquet_paths)} parquets, {total_frames} frames total.")

    # ── Videos ─────────────────────────────────────────────────────
    src_vid_dir = src / "videos" / "chunk-000" / "observation.images.ego_view"
    dst_vid_dir = dst / "videos" / "chunk-000" / "observation.images.ego_view"
    dst_vid_dir.mkdir(parents=True, exist_ok=True)
    n_vids = 0
    for mp4 in sorted(src_vid_dir.glob("episode_*.mp4")):
        dst_mp4 = dst_vid_dir / mp4.name
        if dst_mp4.exists() or dst_mp4.is_symlink():
            dst_mp4.unlink()
        if args.symlink_videos:
            dst_mp4.symlink_to(mp4)
        else:
            shutil.copy2(mp4, dst_mp4)
        n_vids += 1
    print(f"[convert]   {'symlinked' if args.symlink_videos else 'copied'} {n_vids} videos.")

    # ── Meta ───────────────────────────────────────────────────────
    print("[convert] Writing meta...")
    src_info = json.loads((src / "meta" / "info.json").read_text())
    write_meta(src / "meta", dst / "meta",
               num_episodes=len(parquet_paths),
               num_frames=total_frames,
               fps=float(src_info.get("fps", 20.0)),
               max_l=max_l, max_r=max_r,
               empty_hand_proprio=args.empty_hand_proprio)

    # ── Quick sanity check on the first output parquet ─────────────
    out_first = dst_data_dir / parquet_paths[0].name
    df = pd.read_parquet(out_first)
    s0 = np.asarray(df['observation.state'].values[0], dtype=np.float32)
    a0 = np.asarray(df['action'].values[0], dtype=np.float32)
    print()
    print(f"[convert] sanity check on {out_first.name}:")
    print(f"  state.shape  = ({len(df)}, {s0.shape[0]})    expected (T, 28)")
    print(f"  action.shape = ({len(df)}, {a0.shape[0]})    expected (T, 18)")
    print(f"  first-frame state[7]  (left grip)  = {s0[7]:.3f}   (expect ~0 for open hand)")
    print(f"  first-frame state[21] (right grip) = {s0[21]:.3f}  (expect ~0 at episode start)")
    print(f"  first-frame action[17] (r_trig)    = {a0[17]:.3f}  (expect ~0 at episode start)")
    last_a = np.asarray(df['action'].values[-1], dtype=np.float32)
    print(f"  last-frame  action[17] (r_trig)    = {last_a[17]:.3f}  (expect higher, hand closed at grasp)")

    print()
    print(f"[convert] Done. New dataset at: {dst}")


if __name__ == "__main__":
    main()
