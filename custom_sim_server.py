"""Custom GR00T inference server for the robocasa-gr1-tabletop-tasks pipeline.

Bypasses `Gr00tSimPolicyWrapper` (the docstring says "if you are using other
environments, custom robots, or building new environments, you should use
`Gr00tPolicy` directly and format your observations according to its
interface"). This wrapper implements the specific state/action conversion
that matches the finetuning schema documented in
`Deploy_gr00t_sim/convert_sim_to_no_legs.py`.

SCHEMA (must match training)
----------------------------
STATE  (4 modality keys, 7 dims each, 28 total):
    left_arm[7]   ← 7-DOF sim arm joints (passthrough)
    right_arm[7]  ← 7-DOF sim arm joints (passthrough)
    left_hand[7]  ← [grip_scalar, hand_0, hand_1, ..., hand_5]
                     grip = clip(mean(|hand_6|) / max_l, 0, 1); dead-band at 0.1 rad
                     hand_i = 6-DOF Fourier finger joints, OR zeros if empty_hand_proprio
    right_hand[7] ← same idea (uses max_r)

ACTION (2 modality keys, 18 dims total):
    upper[17]     ← waist(3) + L_arm(7) + R_arm(7) reshuffled into PLANNER order
    r_trig[1]     ← analog right-hand closure scalar in [0, 1]

We invert both directions:

    Env state (6-DOF hands)     →   State passed to model (7-DOF hands with grip)
    Model action (upper, r_trig) →  Env action (arm/hand/waist per-part)

The LEFT hand action is emitted as all zeros (model was trained to only close
the right hand; the pick-up task uses the right hand). The right hand action
is `r_trig * grasp_shape` where `grasp_shape` is a canonical 6-DOF fully-closed
pose (default all-1.0, tuneable).

Usage
-----
    python custom_sim_server.py \\
        --model-path ../checkpoint-60000-red-ball-large-sim/ \\
        --embodiment-tag NEW_EMBODIMENT --device cuda:0 --port 5555 \\
        --grip-max-left 0.5 --grip-max-right 0.5 \\
        --language-override "pick up the red cube"

    # ↑ replace 0.5 with the max values recorded in your training dataset's
    #   meta/info.json under `conversion.grip_normalization.max_mean_abs_*`.
"""

import atexit
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import PolicyWrapper
from gr00t.policy.server_client import PolicyServer


# ─── Env flat keys ──────────────────────────────────────────────────
ENV_VIDEO_KEY = "video.ego_view_pad_res256_freq20"
ENV_STATE_LEFT_ARM   = "state.left_arm"
ENV_STATE_RIGHT_ARM  = "state.right_arm"
ENV_STATE_LEFT_HAND  = "state.left_hand"
ENV_STATE_RIGHT_HAND = "state.right_hand"
ENV_LANGUAGE_KEY = "annotation.human.coarse_action"
LANGUAGE_PREFIXES = ("locked_waist: ", "unlocked_waist: ")

# ─── Model modality keys (must match training schema) ───────────────
MODEL_VIDEO_KEY = "ego_view"
MODEL_LANGUAGE_KEY = "annotation.human.action.task_description"

# ─── Constants from convert_sim_to_no_legs.py ───────────────────────
# planner[i] = position in the logical [waist(3), L_arm(7), R_arm(7)] concat
PLANNER_FROM_LOGICAL = [
    0, 1, 2, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16,
]
# Inverse: given planner-order index, where does that value sit in the
# logical concat? Used to split model's `upper` back into waist/L_arm/R_arm.
LOGICAL_FROM_PLANNER = [PLANNER_FROM_LOGICAL.index(i) for i in range(17)]

GRIP_DEAD_BAND = 0.1  # radians — below this, mean(|hand|) is treated as noise


def _hand_state_6_to_7(
    hand: np.ndarray,
    max_grip: float,
    empty_hand_proprio: bool,
) -> np.ndarray:
    """Convert 6-DOF Fourier hand state into the 7-DOF training layout.

    Input : (..., 6) float
    Output: (..., 7) float32 = [grip, hand_0, ..., hand_5]
            OR [grip, 0, 0, 0, 0, 0, 0] if empty_hand_proprio=True
    """
    hand = np.asarray(hand, dtype=np.float32)
    mean_abs = np.mean(np.abs(hand), axis=-1)  # (...,)
    if max_grip <= GRIP_DEAD_BAND:
        grip = np.zeros_like(mean_abs, dtype=np.float32)
    else:
        grip = np.clip(mean_abs / max_grip, 0.0, 1.0).astype(np.float32)
    grip = grip[..., None]  # (..., 1)

    if empty_hand_proprio:
        rest = np.zeros(hand.shape[:-1] + (6,), dtype=np.float32)
    else:
        rest = hand
    return np.concatenate([grip, rest], axis=-1)  # (..., 7)


class CustomSimWrapper(PolicyWrapper):
    """Bridge the flat env obs to nested Gr00tPolicy obs, and split the
    model's (upper, r_trig) action back into per-part actions the env
    expects (arm/hand/waist)."""

    def __init__(
        self,
        policy: Gr00tPolicy,
        *,
        strict: bool = True,
        language_override: str | None = None,
        grip_max_left: float = 1.0,
        grip_max_right: float = 1.0,
        empty_hand_proprio: bool = False,
        right_hand_grasp_shape: np.ndarray | None = None,
        record_attention_dir: str | None = None,
    ):
        super().__init__(policy, strict=strict)
        self.policy = policy
        self.language_override = language_override
        self.grip_max_left = grip_max_left
        self.grip_max_right = grip_max_right
        self.empty_hand_proprio = empty_hand_proprio
        if right_hand_grasp_shape is None:
            right_hand_grasp_shape = np.ones(6, dtype=np.float32)
        self.right_hand_grasp_shape = np.asarray(
            right_hand_grasp_shape, dtype=np.float32
        )
        assert self.right_hand_grasp_shape.shape == (6,)
        self._dumped_shapes = False

        # Introspect what the model actually declares — helpful sanity print.
        try:
            state_cfg = policy.modality_configs["state"]
            action_cfg = policy.modality_configs["action"]
            print(f"[wrapper] model state modality_keys : {list(state_cfg.modality_keys)}")
            print(f"[wrapper] model action modality_keys: {list(action_cfg.modality_keys)}")
        except Exception as e:
            print(f"[wrapper] could not introspect model modality configs: {e}")

        print(
            f"[wrapper] grip_max=(L:{grip_max_left}, R:{grip_max_right}), "
            f"empty_hand_proprio={empty_hand_proprio}, "
            f"right_hand_grasp_shape={self.right_hand_grasp_shape.tolist()}"
        )

        # ── Attention capture (mirrors deploy_groot.py) ──
        # Same idiom as the real-robot deploy: attach a CaptureHandle to the
        # DiT once, then reset()+read() per inference call.  Per-episode
        # bookkeeping is layered on top:
        #   - `_slot_last_state[i]` = last observed concatenated state for
        #     batch slot i.  A large L2 jump vs. the incoming state signals
        #     that slot i has been reset (new episode).
        #   - `_slot_ep_id[i]` = the global episode id currently assigned to
        #     slot i.  Starts at None; every fresh assignment (first call
        #     seen for that slot, or any subsequent reset) bumps
        #     `_next_global_ep_id`.
        #   - `_episode_meta[ep_id]` accumulates start/end timestamp + which
        #     slot ran it, dumped as episodes.json at shutdown.
        # At shutdown, chunks are split by (slot, ep_id) and one .npz is
        # written per episode.
        self.capture_handle = None
        self.record_attention_dir: Path | None = None
        self._chunks: list[dict] = []
        self._call_idx: int = 0
        self._run_start_ts: float = time.time()

        # Per-slot reset detection state (lazy-init on first call once we
        # know the batch size).
        self._slot_last_state: list[np.ndarray | None] | None = None
        self._slot_ep_id: list[int | None] | None = None
        self._next_global_ep_id: int = 0
        self._episode_meta: dict[int, dict] = {}
        # An L2 jump of >0.5 rad across the whole 28-D state between
        # consecutive control steps is huge; real controlled transitions
        # sit well under 0.2 rad at 20 Hz. 0.5 is a comfortable margin.
        self._reset_threshold: float = 0.5

        if record_attention_dir is not None:
            self._init_attention_capture(policy, record_attention_dir)

    def _init_attention_capture(self, policy: Gr00tPolicy, out_dir: str) -> None:
        try:
            from capture_attention import attach as _attach_capture
        except ImportError as e:
            raise RuntimeError(
                "record_attention_dir set but capture_attention.py isn't importable. "
                "Copy creo-g1-teleop/capture_attention.py to a directory on PYTHONPATH."
            ) from e
        try:
            dit = policy.model.action_head.model
        except AttributeError as e:
            raise RuntimeError(
                "Could not find policy.model.action_head.model — expected the DiT "
                "(or AlternateVLDiT). Check the checkpoint's model layout."
            ) from e
        self.capture_handle = _attach_capture(dit)

        # One directory per server run so consecutive launches don't collide.
        run_id = time.strftime("%Y%m%d_%H%M%S")
        self.record_attention_dir = Path(out_dir) / f"run_{run_id}"
        self.record_attention_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self._save_attention)  # flush even on normal shutdown

        print(
            f"[wrapper] attention capture attached — {len(self.capture_handle.cross_block_indices)} "
            f"cross blocks, {len(self.capture_handle.self_block_indices)} self blocks; "
            f"writing to {self.record_attention_dir}"
        )

    # ───── Observation validation ─────
    def check_observation(self, observation: dict[str, Any]) -> None:
        required = [
            ENV_VIDEO_KEY,
            ENV_STATE_LEFT_ARM, ENV_STATE_RIGHT_ARM,
            ENV_STATE_LEFT_HAND, ENV_STATE_RIGHT_HAND,
        ]
        for k in required:
            assert k in observation, (
                f"Env observation missing '{k}'. Available keys: "
                f"{sorted(observation.keys())}"
            )
        if self.language_override is None:
            assert ENV_LANGUAGE_KEY in observation, (
                f"Env observation missing '{ENV_LANGUAGE_KEY}'. "
                f"Pass --language-override to hardcode a task description."
            )

    # ───── Main action pipeline ─────
    def _get_action(self, observation: dict[str, Any], options=None):
        # Build nested obs.
        nested: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
        nested["video"][MODEL_VIDEO_KEY] = observation[ENV_VIDEO_KEY]

        nested["state"]["left_arm"]  = np.asarray(observation[ENV_STATE_LEFT_ARM],  dtype=np.float32)
        nested["state"]["right_arm"] = np.asarray(observation[ENV_STATE_RIGHT_ARM], dtype=np.float32)
        nested["state"]["left_hand"]  = _hand_state_6_to_7(
            observation[ENV_STATE_LEFT_HAND],  self.grip_max_left,  self.empty_hand_proprio,
        )
        nested["state"]["right_hand"] = _hand_state_6_to_7(
            observation[ENV_STATE_RIGHT_HAND], self.grip_max_right, self.empty_hand_proprio,
        )

        # Language: (B, T=1) list[list[str]]
        batch_size = self._infer_batch_size(observation)
        if self.language_override is not None:
            texts = [self.language_override] * batch_size
        else:
            raw = observation[ENV_LANGUAGE_KEY]
            if isinstance(raw, str):
                raw = [raw] * batch_size
            texts = [self._strip_prefix(t) for t in raw]
        nested["language"][MODEL_LANGUAGE_KEY] = [[t] for t in texts]

        # One-shot shape dump — after all nested keys are populated.
        if not self._dumped_shapes:
            self._dump_shapes(nested)
            self._dumped_shapes = True

        # Per-slot reset detection.  Compute a compact per-slot state vector
        # (concatenate arms + hands) and compare to the last one we saw.
        # A large L2 jump means that slot has been reset — reassign its
        # episode id and stamp the previous episode's end time.
        slot_states = np.concatenate(
            [
                nested["state"]["left_arm"],
                nested["state"]["right_arm"],
                nested["state"]["left_hand"],
                nested["state"]["right_hand"],
            ],
            axis=-1,
        )  # (B, D)
        slot_ep_id_this_call = self._assign_episode_ids(slot_states)

        # Inference — bracket with attention capture reset/read.
        if self.capture_handle is not None:
            self.capture_handle.reset()

        action, info = self.policy.get_action(nested, options)

        if self.capture_handle is not None:
            captured = self.capture_handle.read()
            self._record_chunk(
                nested["video"][MODEL_VIDEO_KEY],
                captured,
                slot_ep_ids=slot_ep_id_this_call,
            )

        # Model returns action["upper"] (B, T, 17) and action["r_trig"] (B, T, 1).
        if "upper" not in action or "r_trig" not in action:
            raise RuntimeError(
                f"Model action does not contain expected keys 'upper' and 'r_trig'. "
                f"Got: {list(action.keys())}"
            )

        upper = np.asarray(action["upper"], dtype=np.float32)   # (B, T, 17) planner-order
        r_trig = np.asarray(action["r_trig"], dtype=np.float32)  # (B, T, 1)

        # Split `upper` (planner-order) back into logical [waist(3), L_arm(7), R_arm(7)].
        logical_17 = upper[..., LOGICAL_FROM_PLANNER]  # (B, T, 17)
        waist    = logical_17[..., 0:3]     # (B, T, 3)
        left_arm  = logical_17[..., 3:10]   # (B, T, 7)
        right_arm = logical_17[..., 10:17]  # (B, T, 7)

        # r_trig → 6-DOF right-hand finger target.
        # Squeeze the trailing 1, then broadcast against the grasp shape.
        r_trig_scalar = r_trig[..., 0:1]                          # (B, T, 1)
        right_hand = r_trig_scalar * self.right_hand_grasp_shape  # (B, T, 6)

        # Left hand: model wasn't trained to actuate it — keep open (zeros).
        left_hand = np.zeros_like(right_hand)  # (B, T, 6)

        # Emit env action keys per key_converter.unmap_action.
        flat_action = {
            "action.left_arm":   left_arm,
            "action.right_arm":  right_arm,
            "action.left_hand":  left_hand,
            "action.right_hand": right_hand,
            "action.waist":      waist,
        }

        # Dump action shapes ONCE too (right after inference completes).
        if getattr(self, "_dumped_action_shapes", False) is False:
            print("[wrapper] --- first action: shape dump ---")
            for k, v in flat_action.items():
                print(f"[wrapper]   {k}: shape={v.shape}, dtype={v.dtype}")
            print(f"[wrapper]   raw model action keys : {list(action.keys())}")
            for k, v in action.items():
                arr = np.asarray(v)
                print(f"[wrapper]     raw model action.{k}: shape={arr.shape}, dtype={arr.dtype}")
            print("[wrapper] --- end action dump ---")
            self._dumped_action_shapes = True

        return flat_action, info

    def check_action(self, action: dict[str, Any]) -> None:
        pass  # trust our own construction

    # ───── helpers ─────
    def _infer_batch_size(self, observation: dict[str, Any]) -> int:
        v = observation.get(ENV_VIDEO_KEY)
        return len(v) if v is not None else 1

    @staticmethod
    def _strip_prefix(text):
        if not isinstance(text, str):
            return text
        for prefix in LANGUAGE_PREFIXES:
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    # ───── attention capture ─────
    def _assign_episode_ids(self, slot_states: np.ndarray) -> np.ndarray:
        """Return the global episode id for each of the B slots on this call.

        On the first call, every slot gets a fresh id.  On later calls,
        any slot whose state jumps by more than `_reset_threshold` in L2
        gets a fresh id (and the previous episode's end timestamp is
        stamped in `_episode_meta`).
        """
        B = slot_states.shape[0]
        now = time.time() - self._run_start_ts

        if self._slot_last_state is None:
            self._slot_last_state = [None] * B
            self._slot_ep_id = [None] * B
        elif len(self._slot_last_state) != B:
            # Batch size changed unexpectedly — reset from scratch.  Shouldn't
            # happen in normal rollouts, but be defensive.
            print(
                f"[wrapper] batch size changed {len(self._slot_last_state)} -> {B}; "
                f"reinitializing per-slot state"
            )
            self._slot_last_state = [None] * B
            self._slot_ep_id = [None] * B

        out = np.empty(B, dtype=np.int32)
        for i in range(B):
            prev = self._slot_last_state[i]
            is_reset = prev is None or (
                np.linalg.norm(slot_states[i] - prev) > self._reset_threshold
            )
            if is_reset:
                # Close out the previous episode on this slot, if any.
                if self._slot_ep_id[i] is not None:
                    self._episode_meta[self._slot_ep_id[i]]["end_ts"] = now
                # Open a new episode on this slot.
                new_ep = self._next_global_ep_id
                self._next_global_ep_id += 1
                self._slot_ep_id[i] = new_ep
                self._episode_meta[new_ep] = {
                    "episode_id": new_ep,
                    "slot": i,
                    "start_ts": now,
                    "end_ts": None,  # patched at next reset or at shutdown
                    "n_calls": 0,
                }
            out[i] = self._slot_ep_id[i]
            self._episode_meta[self._slot_ep_id[i]]["n_calls"] += 1
            self._slot_last_state[i] = slot_states[i].copy()
        return out

    def _record_chunk(
        self,
        video_batch: np.ndarray,
        captured: dict,
        slot_ep_ids: np.ndarray,
    ) -> None:
        """Store one inference's capture. `video_batch` shape: (B, T, H, W, 3)."""
        # Take latest frame per env → (B, H, W, 3) uint8.
        img = np.asarray(video_batch)
        if img.ndim == 5:
            img = img[:, -1]
        self._chunks.append({
            "call_idx": self._call_idx,
            "timestamp": time.time() - self._run_start_ts,
            "image": img.astype(np.uint8, copy=False),
            "cross": captured.get("cross"),
            "self":  captured.get("self"),
            "hidden": captured.get("hidden_states"),
            "cross_block_indices": captured.get("cross_block_indices", []),
            "self_block_indices":  captured.get("self_block_indices", []),
            "all_block_indices":   captured.get("all_block_indices", []),
            # Per-slot episode id assigned by _assign_episode_ids — one entry
            # per batch slot, dtype int32, shape (B,).
            "slot_ep_ids": slot_ep_ids.copy(),
        })
        self._call_idx += 1

    def _save_attention(self) -> None:
        """Flush self._chunks into per-episode npz files.

        Each episode E is identified by a `(slot, episode_id)` pair — the id
        was assigned by `_assign_episode_ids` on the first call after that
        slot was reset.  For a given E we walk the recorded chunks and, for
        each call in which slot S had episode id E, extract that slot's
        slice of image + attention tensors and stack them along a new
        time axis.

        Output layout under `self.record_attention_dir`:
            episode_0000.npz
            episode_0001.npz
            ...
            episodes.json       # id -> {slot, start_ts, end_ts, n_calls, npz}
            meta.json           # run-level config (grip max, language override)
        """
        if not self._chunks or self.record_attention_dir is None:
            return

        # Close out any episodes that were still "open" when the server died.
        now = time.time() - self._run_start_ts
        if self._slot_ep_id is not None:
            for ep_id in self._slot_ep_id:
                if ep_id is not None and self._episode_meta.get(ep_id, {}).get("end_ts") is None:
                    self._episode_meta[ep_id]["end_ts"] = now

        # Constant across episodes:
        cross_block_indices = np.array(
            self._chunks[0]["cross_block_indices"], dtype=np.int32
        )
        self_block_indices = np.array(
            self._chunks[0]["self_block_indices"], dtype=np.int32
        )
        hidden_block_indices = np.array(
            self._chunks[0]["all_block_indices"], dtype=np.int32
        )
        image_mask = None
        if self.capture_handle is not None and self.capture_handle.image_mask is not None:
            image_mask = self.capture_handle.image_mask.numpy().astype(np.bool_)

        # Group chunks by episode. For each episode we need a list of
        # (chunk, slot_within_chunk) pointers.
        episode_hits: dict[int, list[tuple[int, int]]] = {}
        for chunk_idx, c in enumerate(self._chunks):
            for slot, ep_id in enumerate(c["slot_ep_ids"]):
                episode_hits.setdefault(int(ep_id), []).append((chunk_idx, slot))

        episodes_json: list[dict] = []
        for ep_id in sorted(episode_hits.keys()):
            hits = episode_hits[ep_id]
            meta = self._episode_meta.get(ep_id, {"slot": hits[0][1]})
            slot = meta["slot"]

            # Build per-time-step arrays for this episode by slicing the
            # right slot out of each hit chunk.
            call_indices = np.array(
                [self._chunks[ci]["call_idx"] for ci, _ in hits], dtype=np.int32
            )
            timestamps = np.array(
                [self._chunks[ci]["timestamp"] for ci, _ in hits], dtype=np.float64
            )
            images = np.stack(
                [self._chunks[ci]["image"][s] for ci, s in hits], axis=0
            )  # (T, H, W, 3)

            payload: dict[str, Any] = {
                "episode_id": np.int32(ep_id),
                "slot": np.int32(slot),
                "call_idx": call_indices,
                "timestamps": timestamps,
                "images": images,
            }

            def _slice_or_none(key: str, slot_axis: int):
                vals = [self._chunks[ci][key] for ci, _ in hits]
                if not all(v is not None for v in vals):
                    return None
                try:
                    per_call_slot = []
                    for (ci, s), v in zip(hits, vals):
                        # v is a torch.Tensor; select this call's slot along
                        # `slot_axis` and drop that axis.
                        per_call_slot.append(v.select(dim=slot_axis, index=s))
                    return torch.stack(per_call_slot, dim=0).numpy()
                except Exception as e:
                    print(f"[wrapper] failed to slice '{key}' for ep {ep_id}: {e}")
                    return None

            # cross shape per call: (n_denoise, n_cross_blocks, B, H, T_q, T_k_vlm)
            # → slot axis is 2.  After slice+stack: (T, n_denoise, n_cross_blocks, H, T_q, T_k_vlm)
            cross = _slice_or_none("cross", slot_axis=2)
            if cross is not None:
                payload["attentions"] = cross
                payload["block_indices"] = cross_block_indices

            self_att = _slice_or_none("self", slot_axis=2)
            if self_att is not None:
                payload["self_attentions"] = self_att
                payload["self_block_indices"] = self_block_indices

            # hidden shape per call: (n_denoise, n_blocks+1, B, T_q, D) → slot axis 2
            hidden = _slice_or_none("hidden", slot_axis=2)
            if hidden is not None:
                payload["hidden_states"] = hidden
                payload["hidden_block_indices"] = hidden_block_indices

            if image_mask is not None:
                payload["image_mask"] = image_mask

            fname = f"episode_{ep_id:04d}.npz"
            out_path = self.record_attention_dir / fname
            np.savez(out_path, **payload)

            episodes_json.append({
                "episode_id": ep_id,
                "slot": int(slot),
                "start_ts": meta.get("start_ts"),
                "end_ts": meta.get("end_ts"),
                "n_calls": meta.get("n_calls", len(hits)),
                "npz": fname,
            })

        # Global manifest so a downstream pairing script can find each npz
        # + video without re-reading the individual files.
        import json
        (self.record_attention_dir / "episodes.json").write_text(
            json.dumps(episodes_json, indent=2)
        )

        # Run-level config (unchanged from before, just no per-call tensor
        # shapes since those now live in the per-episode npzs).
        meta = {
            "n_calls_total": len(self._chunks),
            "n_episodes": len(episodes_json),
            "run_start": self._run_start_ts,
            "run_end": time.time(),
            "grip_max_left": self.grip_max_left,
            "grip_max_right": self.grip_max_right,
            "empty_hand_proprio": self.empty_hand_proprio,
            "language_override": self.language_override,
            "reset_threshold_l2": self._reset_threshold,
        }
        (self.record_attention_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Also detach so the model is left clean if the process keeps living
        # for any reason (e.g. an atexit chain).
        try:
            if self.capture_handle is not None:
                self.capture_handle.detach()
        except Exception as e:
            print(f"[wrapper] capture_handle.detach failed: {e}")

        print(
            f"[wrapper] saved {len(episodes_json)} per-episode npz files "
            f"({len(self._chunks)} calls total) to {self.record_attention_dir}"
        )

    def _dump_shapes(self, nested: dict[str, dict[str, Any]]) -> None:
        print("[wrapper] --- first obs: shape dump ---")
        for mod in ("video", "state"):
            for k, v in nested[mod].items():
                arr = np.asarray(v)
                print(f"[wrapper]   {mod}.{k}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"[wrapper]   language[{MODEL_LANGUAGE_KEY}]: "
              f"{nested['language'][MODEL_LANGUAGE_KEY]!r}")
        print("[wrapper] --- end obs dump ---")


@dataclass
class ServerConfig:
    model_path: str
    """Path to the model checkpoint directory."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag matching the checkpoint's training config."""

    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 5555

    strict: bool = True

    language_override: str | None = None
    """If set, sent to the VLA regardless of what the env's language field is."""

    # ─── Grip normalization (MUST match training data's info.json) ───
    grip_max_left: float = 1.0
    """max(mean(|left_hand_state|)) recorded during dataset conversion.
    Look in your training dataset's meta/info.json under
    conversion.grip_normalization.max_mean_abs_left_hand."""

    grip_max_right: float = 1.0
    """Same idea for the right hand."""

    empty_hand_proprio: bool = False
    """Whether the training dataset was converted with --empty-hand-proprio.
    If True, hand state slot[1:7] are zeroed (only the grip scalar is sent)."""

    right_hand_grasp_shape: str = "1,1,1,1,1,1"
    """Comma-separated 6 floats — the canonical fully-closed pose of the
    right hand. The model's r_trig ∈ [0,1] scalar is multiplied by this
    to produce a 6-DOF finger command. Default is all-1.0; tune based on
    inspecting an actual grasp in your dataset."""

    record_attention_dir: str | None = None
    """If set, attach a CaptureHandle to the DiT and dump attention +
    hidden states to <this>/run_YYYYMMDD_HHMMSS/attention.npz on shutdown.
    Requires capture_attention.py on PYTHONPATH (from creo-g1-teleop/)."""


def _parse_grasp_shape(s: str) -> np.ndarray:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 6:
        raise ValueError(f"--right-hand-grasp-shape needs 6 floats, got {len(vals)}: {s}")
    return np.asarray(vals, dtype=np.float32)


def main(cfg: ServerConfig):
    print(f"[server] loading {cfg.model_path} on {cfg.device}")
    policy = Gr00tPolicy(
        embodiment_tag=cfg.embodiment_tag,
        model_path=cfg.model_path,
        device=cfg.device,
        strict=cfg.strict,
    )
    wrapped = CustomSimWrapper(
        policy,
        strict=cfg.strict,
        language_override=cfg.language_override,
        grip_max_left=cfg.grip_max_left,
        grip_max_right=cfg.grip_max_right,
        empty_hand_proprio=cfg.empty_hand_proprio,
        right_hand_grasp_shape=_parse_grasp_shape(cfg.right_hand_grasp_shape),
        record_attention_dir=cfg.record_attention_dir,
    )
    print(f"[server] listening on {cfg.host}:{cfg.port}")
    server = PolicyServer(policy=wrapped, host=cfg.host, port=cfg.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[server] shutdown")
    finally:
        # Explicit flush — atexit also does this, but running it here means
        # any print()s show before the process starts tearing down.
        wrapped._save_attention()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
