#!/usr/bin/env python3
"""
diag_init_pose.py — minimal diagnostic for the sim init-pose problem.

Isolates the controller question from the VLA. We load the env, read
state[0] from a training parquet, send that pose (and action[0]) to the
controller repeatedly, and print whether the measured joints actually
converge. No policy, no attention capture, no chunked inference.

Three modes (pick with --mode):

  introspect   — build env, env.reset(), dump controller config + part
                 controllers + action_spec + measured joints. Don't step.

  drive-state  — repeatedly send state[0] as the env action (clipped /
                 padded per-part via robot.create_action_vector). Use this
                 to test whether the controller can drive to a known pose.

  drive-action — repeatedly send action[0] from the parquet (the EXACT
                 vector the env saw at data collection time, planner-
                 permuted, after running it through model_action_to_env_action).
                 This is the cleanest "does the controller still work
                 the same way it did during training?" check.

Usage:
  python creo-g1-teleop/diag_init_pose.py \\
         --controller-config ../controller_config.json \\
         --dataset ../red_ball_large_sim_no_legs/ \\
         --mode drive-state \\
         --steps 80 --output diag_video.mp4

If --controller-config is omitted, robosuite's default JOINT_POSITION
controller is used (the same one `collect_human_demonstrations.py` loads
with `--controller JOINT_POSITION`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import robosuite
from robosuite import make

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
# `load_controller_config` is at robosuite.load_controller_config in ≤1.4,
# but in robosuite 1.5+ it moved (the GR1 path now goes through composite
# controllers loaded automatically from `robots/default_gr1.json` when
# `controller_configs=None` is passed to make()). Try a few import paths;
# if none work, we leave controller_configs unset at make() time, which
# triggers robosuite's auto-load of the robot's default config.
load_controller_config = None
for _mod, _name in [
    ("robosuite",                                  "load_controller_config"),
    ("robosuite.controllers",                      "load_controller_config"),
    ("robosuite.controllers",                      "load_part_controller_config"),
    ("robosuite.controllers.controller_factory",   "load_controller_config"),
    ("robosuite.controllers.controller_factory",   "load_part_controller_config"),
]:
    try:
        _m = __import__(_mod, fromlist=[_name])
        load_controller_config = getattr(_m, _name)
        break
    except (ImportError, AttributeError):
        continue


# ── Joint conventions (same as deploy_groot_sim_humandemo.py) ───────
RIGHT_ARM_JOINTS = [
    "robot0_r_shoulder_pitch", "robot0_r_shoulder_roll",
    "robot0_r_shoulder_yaw",   "robot0_r_elbow_pitch",
    "robot0_r_wrist_yaw",      "robot0_r_wrist_roll", "robot0_r_wrist_pitch",
]
LEFT_ARM_JOINTS = [
    "robot0_l_shoulder_pitch", "robot0_l_shoulder_roll",
    "robot0_l_shoulder_yaw",   "robot0_l_elbow_pitch",
    "robot0_l_wrist_yaw",      "robot0_l_wrist_roll", "robot0_l_wrist_pitch",
]
# PLANNER_FROM_LOGICAL[planner_idx] gives the logical slot that the value at
# planner position `planner_idx` belongs to. i.e. planner[i] = logical[PFL[i]].
# Used directly in action18_to_arms to un-interleave the planner-permuted
# action vector back into [waist(3), L_arm(7), R_arm(7)] logical order.
PLANNER_FROM_LOGICAL = [
    0, 1, 2,
    3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16,
]

ARM_PART_NAMES = {"right", "right_arm", "left", "left_arm"}
GRIPPER_PART_NAMES = {"right_gripper", "left_gripper", "right_hand", "left_hand"}

# Dex3 6-DOF gripper qpos indices and pick-order — verbatim from
# collect_data_with_groot.py. Used to read current finger angles so we can
# compute delta-mode gripper commands (robosuite 1.5+ keeps the gripper
# controller in delta mode regardless of `control_delta: false` in the JSON,
# so we have to convert absolute targets to deltas in user code).
QPOS_INDICES_RIGHT_HAND = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
QPOS_INDICES_LEFT_HAND  = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
HAND_DOF_PICK = [0, 1, 4, 6, 8, 10]

# Per-finger reference poses (from collect_data_with_groot.py:236-243).
# Layout matches what the gripper action vector expects after `[::-1]` in
# build_env_action — the same convention as state.right_hand[1:7] in the
# LeRobot dataset.
HAND_OPEN   = np.array([0.003, 0.003, 0.000, 0.000, 0.000, 0.000], dtype=np.float32)
HAND_CLOSED = np.array([0.834, 0.399, 0.416, 0.425, 0.344, 0.256], dtype=np.float32)


def _hand_qpos(env, qpos_indices):
    """Read the 6 actuated dex3 finger angles in the convention used by
    state.right_hand / HAND_OPEN / HAND_CLOSED."""
    full = np.array(env.sim.data.qpos[qpos_indices], dtype=np.float32)
    return full[HAND_DOF_PICK][::-1]


def arm_qpos(env, joint_names):
    return np.array(
        [env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(n)] for n in joint_names],
        dtype=np.float32,
    )


def build_env(args):
    """Build the robosuite env.

    Controller resolution order:
      1. If --controller-config is given → load that JSON.
      2. Else if robosuite exposes a load_controller_config helper → use it.
      3. Else (robosuite 1.5+ default path) → pass controller_configs=None to
         robosuite.make(), which auto-loads the robot's default config
         (e.g. robots/default_gr1.json for GR1ArmsOnly).
    """
    cc = None
    if args.controller_config:
        cc_path = Path(args.controller_config)
        with open(cc_path) as f:
            cc = json.load(f)
        print(f"[diag] controller: custom JSON {cc_path}")
    elif load_controller_config is not None:
        cc = load_controller_config(default_controller=args.controller_type)
        print(f"[diag] controller: robosuite default {args.controller_type}")
    else:
        print(f"[diag] controller: robosuite auto-default for {args.robot} "
              "(passing controller_configs=None to make())")

    if cc is not None:
        print(f"[diag] controller config dump:\n{json.dumps(cc, indent=2)}")

    env = make(
        args.env,
        args.robot,
        controller_configs=cc,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=int(args.fps),
        use_object_obs=True,
        camera_names=args.camera,
        camera_heights=args.image_height,
        camera_widths=args.image_width,
        horizon=args.steps + 50,
    )
    return env


def dump_env_intro(env):
    low, high = env.action_spec
    print(f"[diag] env.action_dim = {env.action_dim}")
    print(f"[diag] action_spec.low  = {np.round(np.asarray(low),  3).tolist()}")
    print(f"[diag] action_spec.high = {np.round(np.asarray(high), 3).tolist()}")
    robot = env.robots[0]
    pc = getattr(robot, "part_controllers", None)
    if pc is not None:
        print("[diag] part_controllers:")
        delta_arm = False
        for k, v in pc.items():
            cd = getattr(v, "control_dim", None) or getattr(v, "action_dim", None)
            ct = type(v).__name__
            in_type = getattr(v, "input_type", "?")
            cd_attr = getattr(v, "control_delta", None)
            print(f"   {k:<18} dim={cd}  type={ct}  input_type={in_type}  control_delta={cd_attr}")
            if k in ARM_PART_NAMES and (str(in_type) == "delta" or cd_attr is True):
                delta_arm = True
        if delta_arm:
            print("[diag] !! WARNING: arm controllers are in DELTA mode. The training data's "
                  "action values are absolute joint angles in radians (e.g. -1.58). With a "
                  "delta-mode controller they'll be clipped to [-1, 1] and applied as small "
                  "per-step deltas — the robot WILL NOT reach the target pose in a chunk of 16 "
                  "steps. Use a custom controller_config.json with input_type='absolute' to "
                  "interpret the model's outputs correctly.")
    else:
        print("[diag] no part_controllers attribute (robosuite <1.5)")


def dump_joint_inventory(env):
    """Print every joint's name + qpos addr + current qpos value. Used to
    verify that the QPOS_INDICES_RIGHT_HAND / LEFT_HAND constants (copied
    from the master-thesis robosuite version) still point at the right
    joints in robosuite 1.5+. Also groups joints by keyword so we can
    quickly see 'hand', 'finger', 'thumb', etc.

    Also dumps `env.robots[0].gripper` info if available — that tells us
    which joint names robosuite considers to be the actuated gripper joints.
    """
    print("\n[diag] --- joint inventory ---")
    model = env.sim.model
    data = env.sim.data
    n_joints = model.njnt

    # Full list, but grouped by keyword.
    groups = {"HAND": [], "FINGER/THUMB/INDEX/MIDDLE/RING/PINKY": [],
              "ARM/SHOULDER/ELBOW/WRIST": [], "OTHER": []}
    for jid in range(n_joints):
        name = model.joint_id2name(jid) or f"<jid={jid}>"
        try:
            addr = model.get_joint_qpos_addr(name)
            # get_joint_qpos_addr returns int for hinge/slide, tuple for ball/free
            if isinstance(addr, tuple):
                lo, hi = addr
                qpos_val = data.qpos[lo:hi].tolist()
                addr_str = f"[{lo}:{hi}]"
                qpos_str = str(np.round(qpos_val, 3).tolist())
            else:
                qpos_val = float(data.qpos[addr])
                addr_str = f"[{addr}]"
                qpos_str = f"{qpos_val:+.3f}"
        except Exception as e:
            addr_str = "??"
            qpos_str = f"<err {e}>"

        line = f"  {name:<50s}  qpos{addr_str:<10s}  = {qpos_str}"
        lname = name.lower()
        if "hand" in lname:
            groups["HAND"].append(line)
        elif any(k in lname for k in
                 ("finger", "thumb", "index", "middle", "ring", "pinky")):
            groups["FINGER/THUMB/INDEX/MIDDLE/RING/PINKY"].append(line)
        elif any(k in lname for k in ("shoulder", "elbow", "wrist", "arm")):
            groups["ARM/SHOULDER/ELBOW/WRIST"].append(line)
        else:
            groups["OTHER"].append(line)

    for gname, lines in groups.items():
        if not lines:
            continue
        print(f"\n[diag] === {gname} ({len(lines)} joints) ===")
        for line in lines:
            print(line)

    # Verify our hardcoded QPOS_INDICES against actual joint names.
    print("\n[diag] --- QPOS_INDICES_RIGHT_HAND probe ---")
    for i, addr in enumerate(QPOS_INDICES_RIGHT_HAND):
        try:
            val = float(data.qpos[addr])
        except Exception:
            val = float("nan")
        # Reverse-look up which joint has this qpos addr.
        owner = "??"
        for jid in range(n_joints):
            name = model.joint_id2name(jid) or ""
            try:
                a = model.get_joint_qpos_addr(name)
                if isinstance(a, int) and a == addr:
                    owner = name
                    break
                if isinstance(a, tuple) and a[0] <= addr < a[1]:
                    owner = f"{name}[{addr - a[0]}]"
                    break
            except Exception:
                pass
        print(f"  qpos[{addr}] = {val:+.3f}   owner = {owner}")

    # Same for left hand.
    print("\n[diag] --- QPOS_INDICES_LEFT_HAND probe ---")
    for i, addr in enumerate(QPOS_INDICES_LEFT_HAND):
        try:
            val = float(data.qpos[addr])
        except Exception:
            val = float("nan")
        owner = "??"
        for jid in range(n_joints):
            name = model.joint_id2name(jid) or ""
            try:
                a = model.get_joint_qpos_addr(name)
                if isinstance(a, int) and a == addr:
                    owner = name
                    break
                if isinstance(a, tuple) and a[0] <= addr < a[1]:
                    owner = f"{name}[{addr - a[0]}]"
                    break
            except Exception:
                pass
        print(f"  qpos[{addr}] = {val:+.3f}   owner = {owner}")

    # Also inspect robot.gripper for the joint names it thinks it controls.
    robot = env.robots[0]
    for side in ("right", "left"):
        grippers = getattr(robot, "gripper", None)
        if isinstance(grippers, dict):
            g = grippers.get(side)
        else:
            g = grippers
        if g is None:
            continue
        joint_names = getattr(g, "_joints", None) or getattr(g, "joints", None) \
                      or getattr(g, "actuated_joints", None)
        actuators   = getattr(g, "_actuators", None) or getattr(g, "actuators", None)
        print(f"\n[diag] gripper.{side} class = {type(g).__name__}")
        if joint_names is not None:
            print(f"[diag] gripper.{side} joints ({len(joint_names)}): {list(joint_names)}")
        if actuators is not None:
            print(f"[diag] gripper.{side} actuators ({len(actuators)}): {list(actuators)}")


def load_dataset_state_and_action(args):
    """Return (state_28, action_18) from episode_{args.episode}.parquet, step args.step."""
    p = Path(args.dataset)
    if p.is_dir():
        p = p / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    print(f"[diag] reading {p} step {args.step}")
    df = pd.read_parquet(p)
    s = np.asarray(df["observation.state"].iloc[args.step], dtype=np.float32)
    a = np.asarray(df["action"].iloc[args.step],            dtype=np.float32)
    print(f"[diag] dataset state[step] len={len(s)}")
    print(f"[diag] dataset action[step] len={len(a)}")
    return s, a


def load_dataset_action_trajectory(args):
    """Return action[step:step+args.steps] as a (T, 18) array from the parquet."""
    p = Path(args.dataset)
    if p.is_dir():
        p = p / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    print(f"[diag] reading {p} steps [{args.step}, {args.step + args.steps})")
    df = pd.read_parquet(p)
    actions = []
    end = min(args.step + args.steps, len(df))
    for i in range(args.step, end):
        actions.append(np.asarray(df["action"].iloc[i], dtype=np.float32))
    actions = np.stack(actions, axis=0)
    print(f"[diag] loaded action trajectory: shape {actions.shape}")
    return actions


def state28_to_arms(s28):
    return (
        s28[14:21].astype(np.float32),   # right_arm  (7)
        s28[0:7].astype(np.float32),     # left_arm   (7)
        s28[21:28].astype(np.float32),   # right_hand (7: grip + 6 fingers)
        s28[7:14].astype(np.float32),    # left_hand
    )


def action18_to_arms(a18):
    """Unpack an 18-D dataset action into (rarm, larm, r_trig).

    Layout (from modality.json): action = [upper(17), r_trig(1)]. `upper` is
    [waist(3), L_arm(0), R_arm(0), L_arm(1), R_arm(1), ..., L_arm(6), R_arm(6)]
    — i.e. waist + planner-interleaved arms (planner[i] = logical[PFL[i]]).

    To recover the logical [waist(3), L_arm(7), R_arm(7)] vector we use
    PLANNER_FROM_LOGICAL directly: each planner-position's value goes to the
    logical slot named by PFL[planner_idx].
    """
    upper17 = a18[:17]
    r_trig = float(a18[17])
    logical = np.empty(17, dtype=np.float32)
    for planner_idx in range(17):
        logical[PLANNER_FROM_LOGICAL[planner_idx]] = upper17[planner_idx]
    larm = logical[3:10]
    rarm = logical[10:17]
    return rarm.astype(np.float32), larm.astype(np.float32), r_trig


def _gripper_slot_sizes(env):
    """Return (right_hand_dim, left_hand_dim) such that the totals match the
    env's action_dim. Layout is always [rarm(7), larm(7), rh(N), lh(M)].

    For the default `default_gr1.json` controller: env.action_dim=24,
    so rh+lh=10 → (5, 5).
    For the masterthesis custom JOINT_POSITION config: env.action_dim=26,
    so rh+lh=12 → (6, 6).
    """
    total_hand = env.action_dim - 14
    if total_hand < 0:
        raise ValueError(f"env.action_dim={env.action_dim} < 14 (two 7-DOF arms)")
    rh = total_hand // 2
    lh = total_hand - rh
    return rh, lh


def _gripper_targets_from_r_trig(r_trig):
    """Map r_trig ∈ [0, 1] to per-finger absolute targets via linear interp
    between HAND_OPEN and HAND_CLOSED. r_trig=0 → all fingers fully open;
    r_trig=1 → all fingers at the master-thesis fully-closed pose."""
    t = float(np.clip(r_trig, 0.0, 1.0))
    return HAND_OPEN + t * (HAND_CLOSED - HAND_OPEN)


def _gripper_action_from_targets(env, side, target_rh):
    """Convert an absolute 6-DOF finger target into the action-slot vector
    the gripper controller actually expects.

    For robosuite 1.5+ default GR1, the gripper is reported as JOINT_POSITION
    but is internally locked in DELTA mode (`control_delta: false` in the JSON
    is silently ignored). So we read the current finger angles and emit
    `clip(target - current, -1, 1)` as the action — that drives the joint
    toward `target` at one delta-step per env.step.

    If the gripper does honor absolute mode (e.g. robosuite ≤ 1.4 with the
    masterthesis config), pass target directly: it'll be interpreted as the
    joint target.
    """
    if side == "right":
        qpos_idx = QPOS_INDICES_RIGHT_HAND
        ctrl_key = "right_gripper"
    else:
        qpos_idx = QPOS_INDICES_LEFT_HAND
        ctrl_key = "left_gripper"

    pc = getattr(env.robots[0], "part_controllers", None)
    ctrl = pc.get(ctrl_key) if pc is not None else None
    in_type = str(getattr(ctrl, "input_type", "")).lower() if ctrl is not None else ""
    is_delta = (in_type == "delta") or (getattr(ctrl, "control_delta", None) is True)

    if is_delta:
        current = _hand_qpos(env, qpos_idx)
        return np.clip(target_rh - current, -1.0, 1.0)
    return target_rh.astype(np.float32)


def build_env_action(env, rarm, larm, r_trig=0.0):
    """Build the flat env action vector for the env's actuation layout.

    Robosuite's `actuation_part_names` order is NOT guaranteed to be
    [right_arm, left_arm, right_gripper, left_gripper]. For the default
    `default_gr1.json` config it's actually [right, right_gripper, left,
    left_gripper] — so a manual concat in [rarm, larm, rh, lh] order
    cross-wires the right_gripper slots with the left_arm slots and breaks
    the rollout (the symptom: left arm collapses to zero, right gripper
    receives joint-angle-magnitude commands and goes berserk).

    Preferred path: hand a per-part dict to `robot.create_action_vector`,
    which uses the authoritative `actuation_part_names` ordering and the
    correct per-part action-input dim. Falls back to manual concat for
    robosuite ≤ 1.4 where that API isn't available.

    Gripper convention: `r_trig` broadcast across all right-gripper slots
    (delta in [-1, 1]; positive = close fingers). Left gripper stays at 0.
    """
    rarm = np.asarray(rarm, dtype=np.float32)
    larm = np.asarray(larm, dtype=np.float32)
    r_trig_clipped = float(np.clip(r_trig, -1.0, 1.0))
    rh_dim, lh_dim = _gripper_slot_sizes(env)

    if rh_dim == 6:
        # Custom masterthesis dex3 config (FourierRightHand / FourierLeftHand
        # in robosuite 1.5+). Its JointPositionController wrapper only ever
        # moves the fingers ~0.02 rad no matter what delta we send — see the
        # `gripper-probe` verdict. So we drive the fingers via a direct qpos
        # write in the runner (`apply_gripper_qpos_bypass(env, r_trig)`
        # called before env.step) and send ZEROS in the action-vector gripper
        # slots so the controller doesn't fight us.
        rh = np.zeros(rh_dim, dtype=np.float32)
        lh = np.zeros(lh_dim, dtype=np.float32)
    else:
        # Default GR1 (5-slot) gripper — broadcast r_trig as before.
        rh = np.full(rh_dim, r_trig_clipped, dtype=np.float32)
        lh = np.zeros(lh_dim, dtype=np.float32)

    robot = env.robots[0]
    pc = getattr(robot, "part_controllers", None)
    if pc is not None and hasattr(robot, "create_action_vector"):
        action_dict = {}
        for name in pc.keys():
            if name in ARM_PART_NAMES:
                action_dict[name] = rarm if "right" in name else larm
            elif name in GRIPPER_PART_NAMES:
                action_dict[name] = rh if "right" in name else lh
            else:
                # Unknown part — zero-fill to the slot's nominal control_dim
                # so robosuite's create_action_vector doesn't reject the dict.
                cd = getattr(pc[name], "control_dim", None) or \
                     getattr(pc[name], "action_dim", None) or 1
                action_dict[name] = np.zeros(int(cd), dtype=np.float32)
        try:
            action = np.asarray(robot.create_action_vector(action_dict),
                                dtype=np.float32)
            if action.shape[0] == env.action_dim:
                return action
            print(f"[diag] WARNING: create_action_vector returned "
                  f"{action.shape[0]}-D, expected {env.action_dim}; "
                  "falling back to manual concat")
        except Exception as e:
            print(f"[diag] WARNING: create_action_vector failed ({e}); "
                  "falling back to manual concat")

    # Fallback (robosuite ≤ 1.4 / custom 26-D layout): assume legacy
    # [rarm, larm, rh[::-1], lh[::-1]] ordering.
    action = np.concatenate([rarm, larm, rh[::-1], lh[::-1]]).astype(np.float32)
    if action.shape[0] != env.action_dim:
        raise RuntimeError(
            f"[diag] built {action.shape[0]}-D action but env.action_dim={env.action_dim}"
        )
    return action


def get_frame(env, args):
    obs = env._get_observations()
    key = f"{args.camera}_image"
    if key not in obs:
        return None
    img = np.flipud(obs[key]).copy()
    return img.astype(np.uint8)


def run_gripper_probe(env, args):
    """Diagnostic that isolates the gripper. On every env.step:
      1. Send zero deltas for both arms (arms should hold near reset pose).
      2. Send a MAX-magnitude command to the gripper slots (full close).
      3. Log the qpos delta across the whole mujoco state to find which
         joints actually move in response.

    If nothing moves in the qpos region we assume is the right hand
    (QPOS_INDICES_RIGHT_HAND) but SOMETHING moves elsewhere → our indices
    are stale. If nothing moves anywhere → the gripper controller is
    genuinely inert and we need to bypass it via
    `env.robots[0].set_gripper_joint_positions`.
    """
    n_steps = args.steps
    print(f"\n[diag] === gripper-probe: {n_steps} steps ===")

    # Snapshot arm joint positions at reset so we can send them as the
    # "hold" arm targets (arm controller is absolute-mode so this holds).
    hold_rarm = arm_qpos(env, RIGHT_ARM_JOINTS)
    hold_larm = arm_qpos(env, LEFT_ARM_JOINTS)
    print(f"[diag] arm hold rarm = {np.round(hold_rarm, 3).tolist()}")
    print(f"[diag] arm hold larm = {np.round(hold_larm, 3).tolist()}")

    # Build a full-close gripper action by forcing r_trig=1.0 inside
    # build_env_action (which internally lerps HAND_OPEN → HAND_CLOSED and,
    # for delta-mode grippers, subtracts current qpos to make a delta).
    n_qpos = env.sim.model.nq
    qpos_before_first = np.array(env.sim.data.qpos, dtype=np.float32).copy()
    print(f"[diag] full qpos length = {n_qpos}")
    print(f"[diag] qpos at reset (rounded, first 40): "
          f"{np.round(qpos_before_first[:40], 3).tolist()}")

    # Video setup (same as other modes).
    writer = None
    if args.output:
        fr0 = get_frame(env, args)
        if fr0 is not None:
            h, w = fr0.shape[:2]
            writer = cv2.VideoWriter(
                args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps), (w, h),
            )

    log_every = max(1, n_steps // 10)
    cumulative_delta = np.zeros(n_qpos, dtype=np.float32)
    for t in range(n_steps):
        qpos_before = np.array(env.sim.data.qpos, dtype=np.float32).copy()
        env_action = build_env_action(env, hold_rarm, hold_larm, r_trig=1.0)
        obs, _, done, _ = env.step(env_action)
        qpos_after = np.array(env.sim.data.qpos, dtype=np.float32)
        step_delta = qpos_after - qpos_before
        cumulative_delta += step_delta

        if t == 0 or (t + 1) % log_every == 0 or t == n_steps - 1:
            # Which qpos entries moved by > 1e-4 this step?
            moved = [(i, float(step_delta[i]))
                     for i in range(n_qpos) if abs(step_delta[i]) > 1e-4]
            print(f"[diag] t={t:3d}  moved-this-step (|Δqpos|>1e-4): "
                  f"{len(moved)} joints")
            for i, dv in moved[:30]:
                # Look up joint owner for this qpos index.
                owner = _qpos_owner(env, i)
                print(f"       qpos[{i:3d}] Δ={dv:+.4f}  cumΔ={cumulative_delta[i]:+.4f}  owner={owner}")
        if writer is not None:
            fr = get_frame(env, args)
            if fr is not None:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        if done and not args.ignore_done:
            break

    if writer is not None:
        writer.release()
        print(f"[diag] video saved → {args.output}")

    # Final: which qpos entries moved the most over the whole probe?
    print(f"\n[diag] --- gripper-probe summary ---")
    idx_sorted = np.argsort(-np.abs(cumulative_delta))
    top = idx_sorted[:20]
    print(f"[diag] top 20 qpos slots by |cumulative delta|:")
    for i in top:
        cd = float(cumulative_delta[i])
        if abs(cd) < 1e-4:
            continue
        owner = _qpos_owner(env, int(i))
        print(f"  qpos[{i:3d}]  cumΔ={cd:+.4f}   owner={owner}")

    if float(np.max(np.abs(cumulative_delta))) < 1e-3:
        print("[diag] VERDICT: nothing moved. The gripper controller is inert. "
              "Bypass it via `env.robots[0].set_gripper_joint_positions(...)`.")
    else:
        # Show which joints moved most vs our assumed QPOS_INDICES_RIGHT_HAND
        assumed_moved = float(np.max(np.abs(cumulative_delta[QPOS_INDICES_RIGHT_HAND])))
        print(f"[diag] max |cumΔ| within QPOS_INDICES_RIGHT_HAND = {assumed_moved:.4f}")
        if assumed_moved < 1e-3:
            print("[diag] VERDICT: some joints moved but NOT the ones we're reading. "
                  "QPOS_INDICES_RIGHT_HAND is stale. Update it based on the "
                  "'top 20 qpos' list above (or look up by joint name).")
        else:
            print("[diag] VERDICT: gripper joints ARE moving in the region we expect. "
                  "The problem is elsewhere (maybe too weak / needs bigger r_trig, "
                  "or the finger reversal in build_env_action is off).")


# Mapping from the 6-DOF finger vector (HAND_OPEN/HAND_CLOSED order) to the
# 11-DOF joint layout that qpos[QPOS_INDICES_RIGHT_HAND] expects. Verbatim
# from collect_data_with_groot.py:413.
HAND_6_TO_11_INDICES = np.array([1, 0, 0, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)


def _expand_6_to_11(hand6):
    """Master-thesis 6-to-11 expansion (collect_data_with_groot.py:412-414):
        action_fingers = hand6[::-1]                 # reverse
        new_hand = action_fingers[HAND_6_TO_11_INDICES]
    """
    action_fingers = np.asarray(hand6, dtype=np.float32)[::-1]
    return action_fingers[HAND_6_TO_11_INDICES].astype(np.float32)


def apply_gripper_qpos_bypass(env, r_trig, left_r_trig=0.0):
    """Runner-facing wrapper: compute per-finger target from r_trig, expand
    to 11 joints, write directly to sim.data.qpos, zero qvel there. Should
    be called BEFORE env.step in run_drive / run_replay when the env has
    the custom 6-slot dex3 gripper (rh_dim == 6). Only enabled when
    `_gripper_slot_sizes(env) == (6, 6)`; a no-op otherwise so the default
    GR1 controller path stays untouched.
    """
    try:
        rh_dim, lh_dim = _gripper_slot_sizes(env)
    except ValueError:
        return
    if rh_dim != 6:
        return
    target_rh6 = HAND_OPEN + float(np.clip(r_trig, 0.0, 1.0)) * (HAND_CLOSED - HAND_OPEN)
    target_lh6 = HAND_OPEN + float(np.clip(left_r_trig, 0.0, 1.0)) * (HAND_CLOSED - HAND_OPEN)
    _write_gripper_qpos(env, target_rh6, target_lh6)


def _write_gripper_qpos(env, target_rh6, target_lh6):
    """Directly write the 11 finger qpos slots for both hands and zero their
    qvel. Bypasses the FourierRightHand/FourierLeftHand controller entirely
    (which seems to only be capable of moving joints by ~0.02 rad regardless
    of input).
    """
    target_rh11 = _expand_6_to_11(target_rh6)
    target_lh11 = _expand_6_to_11(target_lh6)
    env.sim.data.qpos[QPOS_INDICES_RIGHT_HAND] = target_rh11
    env.sim.data.qpos[QPOS_INDICES_LEFT_HAND]  = target_lh11
    # Zero the velocities on those joints so the PD sees no velocity error.
    # In mujoco, qvel indices for hinge joints line up 1:1 with qpos indices.
    env.sim.data.qvel[QPOS_INDICES_RIGHT_HAND] = 0.0
    env.sim.data.qvel[QPOS_INDICES_LEFT_HAND]  = 0.0
    env.sim.forward()


def run_gripper_force_qpos(env, args):
    """Bypass test: write the target finger qpos DIRECTLY (via sim.data.qpos)
    before each env.step, with zero gripper action in the env action vector.
    If the fingers visibly close and stay closed → direct-write path works,
    we can integrate it into build_env_action. If they snap back to open
    each step → the PID is fighting us and we need a stronger bypass.
    """
    n_steps = args.steps
    print(f"\n[diag] === gripper-force-qpos: {n_steps} steps ===")

    hold_rarm = arm_qpos(env, RIGHT_ARM_JOINTS)
    hold_larm = arm_qpos(env, LEFT_ARM_JOINTS)

    # Target: fully closed right hand, open left hand.
    target_rh6 = HAND_CLOSED.copy()
    target_lh6 = HAND_OPEN.copy()
    print(f"[diag] target right hand (6-DOF): {target_rh6.tolist()}")
    print(f"[diag] expanded to 11 joints:     {_expand_6_to_11(target_rh6).tolist()}")

    writer = None
    if args.output:
        fr0 = get_frame(env, args)
        if fr0 is not None:
            h, w = fr0.shape[:2]
            writer = cv2.VideoWriter(
                args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps), (w, h),
            )

    log_every = max(1, n_steps // 10)
    for t in range(n_steps):
        # 1. Write target gripper qpos directly.
        _write_gripper_qpos(env, target_rh6, target_lh6)
        rh_after_write = np.array(env.sim.data.qpos[QPOS_INDICES_RIGHT_HAND],
                                   dtype=np.float32).copy()

        # 2. Build env action with zero gripper action (send OPEN so delta=0
        #    after our qpos write). Arms hold their reset pose.
        env_action = build_env_action(env, hold_rarm, hold_larm, r_trig=0.0)
        # Force the gripper slots to zero regardless of what build_env_action did.
        # In the custom controller_config.json layout, the last 12 slots are
        # right_gripper(6) then left_gripper(6).
        env_action = env_action.copy()
        env_action[-12:] = 0.0

        obs, _, done, _ = env.step(env_action)
        rh_after_step = np.array(env.sim.data.qpos[QPOS_INDICES_RIGHT_HAND],
                                  dtype=np.float32)
        if t == 0 or (t + 1) % log_every == 0 or t == n_steps - 1:
            drift = rh_after_step - rh_after_write
            print(f"[diag] t={t:3d}  wrote right qpos = "
                  f"{np.round(rh_after_write, 3).tolist()}")
            print(f"[diag]        after env.step   = "
                  f"{np.round(rh_after_step, 3).tolist()}")
            print(f"[diag]        drift            = "
                  f"{np.round(drift, 4).tolist()}")

        if writer is not None:
            fr = get_frame(env, args)
            if fr is not None:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        if done and not args.ignore_done:
            break

    if writer is not None:
        writer.release()
        print(f"[diag] video saved → {args.output}")

    final_rh = np.array(env.sim.data.qpos[QPOS_INDICES_RIGHT_HAND],
                        dtype=np.float32)
    diff_from_target = final_rh - _expand_6_to_11(target_rh6)
    print(f"\n[diag] --- gripper-force-qpos summary ---")
    print(f"[diag] final right qpos           = {np.round(final_rh, 3).tolist()}")
    print(f"[diag] final |qpos - target_11|   = "
          f"{float(np.max(np.abs(diff_from_target))):.4f}")
    if float(np.max(np.abs(diff_from_target))) < 0.05:
        print("[diag] VERDICT: direct qpos write HOLDS. Bake this into build_env_action.")
    else:
        print("[diag] VERDICT: PID undoes the write. Need `robot.set_gripper_joint_positions` "
              "or set sim.data.ctrl directly.")


def _qpos_owner(env, addr):
    """Reverse-lookup which joint owns qpos slot `addr`. Returns the joint
    name (or 'name[k]' for k-th slot of a multi-DOF joint)."""
    model = env.sim.model
    for jid in range(model.njnt):
        name = model.joint_id2name(jid) or ""
        try:
            a = model.get_joint_qpos_addr(name)
        except Exception:
            continue
        if isinstance(a, int) and a == addr:
            return name
        if isinstance(a, tuple) and a[0] <= addr < a[1]:
            return f"{name}[{addr - a[0]}]"
    return "?"



def run_replay(env, args, action_traj_18):
    """Replay a recorded action trajectory step by step. For each row in
    `action_traj_18` we extract (rarm, larm, r_trig), build the 26-D env
    action, and step the env once. Saves video and prints per-step error
    vs the expected next state (if dataset has next-state info)."""
    writer = None
    if args.output:
        fr0 = get_frame(env, args)
        if fr0 is not None:
            h, w = fr0.shape[:2]
            writer = cv2.VideoWriter(
                args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps), (w, h),
            )

    print(f"\n[diag] === replay-trajectory: {len(action_traj_18)} steps ===")
    log_every = max(1, len(action_traj_18) // 12)
    for t, a18 in enumerate(action_traj_18):
        rarm, larm, r_trig = action18_to_arms(a18)
        env_action = build_env_action(env, rarm, larm, r_trig)
        if t == 0:
            print(f"[diag] action[0] (18-D): {np.round(a18, 3).tolist()}")
            print(f"[diag] target rarm     : {np.round(rarm, 3).tolist()}")
            print(f"[diag] target larm     : {np.round(larm, 3).tolist()}")
            print(f"[diag] r_trig          : {r_trig:+.3f}")
            print(f"[diag] env_action (26-D): {np.round(env_action, 3).tolist()}")
        # Bypass FourierRightHand controller by writing target finger qpos
        # directly. No-op for the default 5-slot gripper.
        apply_gripper_qpos_bypass(env, r_trig)
        obs, _, done, _ = env.step(env_action)
        if t % log_every == 0 or t == len(action_traj_18) - 1:
            ra = arm_qpos(env, RIGHT_ARM_JOINTS)
            la = arm_qpos(env, LEFT_ARM_JOINTS)
            err_r = float(np.max(np.abs(ra - rarm)))
            err_l = float(np.max(np.abs(la - larm)))
            print(f"[diag] t={t:3d}  R_elbow={ra[3]:+.3f} (target {rarm[3]:+.3f})  "
                  f"L_elbow={la[3]:+.3f} (target {larm[3]:+.3f})  "
                  f"max|err|_R={err_r:.3f}  max|err|_L={err_l:.3f}  r_trig={r_trig:+.3f}")
        if writer is not None:
            fr = get_frame(env, args)
            if fr is not None:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        if done and not args.ignore_done:
            print(f"[diag] env signaled done at t={t}")
            break

    if writer is not None:
        writer.release()
        print(f"[diag] video saved → {args.output}")


def run_drive(env, args, target_rarm, target_larm, r_trig, label):
    """Send the same action repeatedly and log measured-joint convergence."""
    writer = None
    if args.output:
        fr0 = get_frame(env, args)
        if fr0 is not None:
            h, w = fr0.shape[:2]
            writer = cv2.VideoWriter(
                args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps), (w, h),
            )

    env_action = build_env_action(env, target_rarm, target_larm, r_trig)
    print(f"\n[diag] === {label} ===")
    print(f"[diag] target right_arm = {np.round(target_rarm, 3).tolist()}")
    print(f"[diag] target left_arm  = {np.round(target_larm, 3).tolist()}")
    print(f"[diag] r_trig           = {r_trig:+.3f}")
    print(f"[diag] env_action ({env_action.shape[0]}-D) = "
          f"{np.round(env_action, 3).tolist()}")

    log_every = max(1, args.steps // 10)
    err_history = []
    for t in range(args.steps):
        # Direct-qpos bypass for the custom dex3 gripper. No-op for default GR1.
        apply_gripper_qpos_bypass(env, r_trig)
        obs, _, done, _ = env.step(env_action)
        ra = arm_qpos(env, RIGHT_ARM_JOINTS)
        la = arm_qpos(env, LEFT_ARM_JOINTS)
        err_r = float(np.max(np.abs(ra - target_rarm)))
        err_l = float(np.max(np.abs(la - target_larm)))
        err_history.append((err_r, err_l))
        if t % log_every == 0 or t == args.steps - 1:
            print(f"[diag] t={t:3d}  max|err|_R={err_r:.3f}  max|err|_L={err_l:.3f}  "
                  f"R_elbow={ra[3]:+.3f}  L_elbow={la[3]:+.3f}")
        if writer is not None:
            fr = get_frame(env, args)
            if fr is not None:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        if done and not args.ignore_done:
            break

    if writer is not None:
        writer.release()
        print(f"[diag] video saved → {args.output}")

    # Final measured pose & summary
    ra = arm_qpos(env, RIGHT_ARM_JOINTS)
    la = arm_qpos(env, LEFT_ARM_JOINTS)
    print(f"[diag] final right_arm  = {np.round(ra, 3).tolist()}")
    print(f"[diag] final left_arm   = {np.round(la, 3).tolist()}")
    print(f"[diag] final |err|_R    = {np.max(np.abs(ra - target_rarm)):.3f}")
    print(f"[diag] final |err|_L    = {np.max(np.abs(la - target_larm)):.3f}")
    if err_history:
        first_err = (err_history[0][0] + err_history[0][1]) / 2
        last_err  = (err_history[-1][0] + err_history[-1][1]) / 2
        if last_err < 0.05:
            print("[diag] VERDICT: converged ✓ (controller drives to the target)")
        elif last_err < first_err * 0.5:
            print(f"[diag] VERDICT: partially converged (err {first_err:.3f} → {last_err:.3f}). "
                  "Try more --steps or check controller gains.")
        else:
            print(f"[diag] VERDICT: NOT converging (err {first_err:.3f} → {last_err:.3f}). "
                  "Likely a controller-mode mismatch (absolute vs delta) or wrong action layout.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="introspect",
                   choices=("introspect", "drive-state", "drive-action",
                            "replay-trajectory", "gripper-probe",
                            "gripper-force-qpos"),
                   help="introspect: just dump env + joint inventory; "
                        "drive-state: HOLD dataset state[step] as the action for --steps; "
                        "drive-action: HOLD dataset action[step] for --steps; "
                        "replay-trajectory: FOLLOW dataset action[step:step+steps]; "
                        "gripper-probe: hold arms + blast a full-close gripper command "
                        "for --steps steps, print the per-step qpos delta so we can "
                        "see which joints actually respond (or don't); "
                        "gripper-force-qpos: bypass gripper controller by writing "
                        "target finger qpos directly to sim.data.qpos each step. "
                        "Tests whether the direct-write path holds.")
    p.add_argument("--env",    default="Lift")
    p.add_argument("--robot",  default="GR1ArmsOnly")
    p.add_argument("--camera", default="frontview")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--image-width",  type=int, default=256)
    p.add_argument("--image-height", type=int, default=256)
    p.add_argument("--controller-config", default=None,
                   help="JSON path. If omitted, use robosuite default.")
    p.add_argument("--controller-type", default="JOINT_POSITION")
    p.add_argument("--dataset", default=None,
                   help="Path to a LeRobot v3 no-legs dataset folder or parquet.")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--step",    type=int, default=0)
    p.add_argument("--steps",   type=int, default=80,
                   help="How many env.step()s to send the target action for.")
    p.add_argument("--output",  default=None,
                   help="If set, save the diagnostic rollout to this mp4.")
    p.add_argument("--ignore-done", action="store_true")
    p.add_argument("--r-trig", type=float, default=None,
                   help="Override the right-gripper command (default: read from dataset).")

    args = p.parse_args()

    print(f"[diag] robosuite {robosuite.__version__}  env={args.env}  robot={args.robot}")
    env = build_env(args)

    print("\n--- BEFORE env.reset() ---")
    # Some robosuite versions populate sim only after reset; skip dump here
    print("\n--- AFTER env.reset() ---")
    env.reset()
    dump_env_intro(env)
    ra = arm_qpos(env, RIGHT_ARM_JOINTS)
    la = arm_qpos(env, LEFT_ARM_JOINTS)
    print(f"[diag] reset right_arm = {np.round(ra, 3).tolist()}")
    print(f"[diag] reset left_arm  = {np.round(la, 3).tolist()}")

    # Joint inventory always runs after reset — this is the diagnostic we
    # really need for the gripper problem. Cheap; adds ~1s to any invocation.
    dump_joint_inventory(env)

    if args.mode == "introspect":
        print("\n[diag] introspect mode — not stepping. Pass --mode drive-state, drive-action, or gripper-probe.")
        env.close()
        return

    if args.mode == "gripper-probe":
        run_gripper_probe(env, args)
        env.close()
        return

    if args.mode == "gripper-force-qpos":
        run_gripper_force_qpos(env, args)
        env.close()
        return

    if args.dataset is None:
        print("[diag] ERROR: --dataset is required for drive modes.", file=sys.stderr)
        sys.exit(2)

    if args.mode == "replay-trajectory":
        traj = load_dataset_action_trajectory(args)
        run_replay(env, args, traj)
    else:
        s28, a18 = load_dataset_state_and_action(args)
        if args.mode == "drive-state":
            rarm, larm, _rh, _lh = state28_to_arms(s28)
            r_trig = args.r_trig if args.r_trig is not None else 0.0
            run_drive(env, args, rarm, larm, r_trig, "drive to state[step]")
        else:  # drive-action
            rarm, larm, r_trig_ds = action18_to_arms(a18)
            r_trig = args.r_trig if args.r_trig is not None else r_trig_ds
            run_drive(env, args, rarm, larm, r_trig, "drive to action[step]")

    env.close()


if __name__ == "__main__":
    main()
