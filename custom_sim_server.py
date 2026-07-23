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
        #   - Each incoming observation carries a `slot_ep_ids: (B,) int32`
        #     array put there by the patched `rollout_policy.py` on the
        #     client.  It's the global episode id currently assigned to
        #     each vec-env slot.  The server just trusts these values —
        #     the client is the ground truth for what an "episode" means.
        #   - `_last_slot_ep_ids[i]` remembers what episode id slot i had
        #     on the previous call, so we can stamp end_ts on the closed
        #     episode when slot i transitions to a new id.
        #   - `_episode_meta[ep_id]` accumulates start/end timestamp,
        #     which slot ran it, and n_calls, dumped as episodes.json
        #     at shutdown.
        # At shutdown, chunks are split by (slot, ep_id) and one .npz is
        # written per episode.
        self.capture_handle = None
        self.record_attention_dir: Path | None = None
        self._chunks: list[dict] = []
        self._call_idx: int = 0
        self._run_start_ts: float = time.time()

        # Per-slot episode tracking. Populated on the first call once we see
        # `slot_ep_ids` in the observation.
        self._last_slot_ep_ids: np.ndarray | None = None
        self._episode_meta: dict[int, dict] = {}

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

        # Per-slot episode ids come from the client — the patched
        # rollout_policy.py stuffs `slot_ep_ids: (B,) int32` into the
        # observation before every `policy.get_action` call.  We read it
        # here and update our meta dict; no discontinuity detection
        # needed.  Fall back to a single always-slot-0 episode if the
        # key is missing (unpatched client) so runs still work — just
        # with all captures grouped as one episode.
        slot_ep_id_this_call = self._read_slot_ep_ids(observation)

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
    def _read_slot_ep_ids(self, observation: dict[str, Any]) -> np.ndarray:
        """Read `slot_ep_ids` from the observation (put there by the patched
        `rollout_policy.py`) and update per-episode meta.  If the key is
        missing (unpatched client), synthesize a run that treats every call
        as belonging to the same single episode 0 — noisy but non-fatal."""
        batch_size = self._infer_batch_size(observation)
        raw = observation.get("slot_ep_ids")
        if raw is None:
            if not getattr(self, "_warned_missing_ep_ids", False):
                print(
                    "[wrapper] WARN: observation has no 'slot_ep_ids' — the "
                    "client isn't patched.  Everything will be grouped as "
                    "one episode.  Apply patch_rollout_policy_episode_ids.sh "
                    "to fix."
                )
                self._warned_missing_ep_ids = True
            slot_ep_ids = np.zeros(batch_size, dtype=np.int32)
        else:
            slot_ep_ids = np.asarray(raw, dtype=np.int32).reshape(-1)
            if slot_ep_ids.size != batch_size:
                raise ValueError(
                    f"slot_ep_ids length {slot_ep_ids.size} does not match "
                    f"batch size {batch_size}"
                )

        now = time.time() - self._run_start_ts

        # First call: register every slot's initial episode.
        if self._last_slot_ep_ids is None:
            self._last_slot_ep_ids = slot_ep_ids.copy()
            for slot, ep_id in enumerate(slot_ep_ids.tolist()):
                self._episode_meta.setdefault(int(ep_id), {
                    "episode_id": int(ep_id),
                    "slot": int(slot),
                    "start_ts": now,
                    "end_ts": None,
                    "n_calls": 0,
                })
        else:
            # Subsequent calls: any slot whose id changed just started a new
            # episode.  Stamp end_ts on the outgoing episode and open a new
            # entry for the incoming one.
            for slot in range(batch_size):
                incoming = int(slot_ep_ids[slot])
                outgoing = int(self._last_slot_ep_ids[slot])
                if incoming != outgoing:
                    if outgoing in self._episode_meta and self._episode_meta[outgoing]["end_ts"] is None:
                        self._episode_meta[outgoing]["end_ts"] = now
                    self._episode_meta.setdefault(incoming, {
                        "episode_id": incoming,
                        "slot": int(slot),
                        "start_ts": now,
                        "end_ts": None,
                        "n_calls": 0,
                    })
            self._last_slot_ep_ids = slot_ep_ids.copy()

        # Bump n_calls for every slot's currently active episode.
        for slot in range(batch_size):
            ep_id = int(slot_ep_ids[slot])
            if ep_id in self._episode_meta:
                self._episode_meta[ep_id]["n_calls"] += 1

        return slot_ep_ids

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
            # Per-slot episode id read from the client's observation — one
            # entry per batch slot, dtype int32, shape (B,).
            "slot_ep_ids": slot_ep_ids.copy(),
        })
        self._call_idx += 1

    def _save_attention(self) -> None:
        """Flush self._chunks into per-episode npz files.

        Save order is deliberately:
          1. `episodes.json` (the manifest) — tiny, always written first,
             marks each episode with `"written": false`.
          2. `meta.json` — run-level config.
          3. Each `episode_XXXX.npz` — huge, may take seconds each.  After
             every successful write, `episodes.json` is atomically
             rewritten with `"written": true` for that episode.

        This means the downstream `pair_videos_to_episodes.py` can pair
        against a partially-written run (or reconstruct from the npz
        files alone if `episodes.json` never landed — it walks the dir
        as a fallback).

        Ctrl-C during the tensor writes is handled: we install a SIGINT
        guard that catches the first interrupt, warns, and lets the
        current npz finish; a second Ctrl-C is honoured and stops early.
        Whatever episodes did land are already recorded in the manifest.
        """
        if not self._chunks or self.record_attention_dir is None:
            return
        # Idempotency guard: atexit + explicit finally will both call us.
        if getattr(self, "_save_done", False):
            return
        self._save_done = True

        import json
        import signal

        # Close out any episodes that were still "open" when the server died.
        now = time.time() - self._run_start_ts
        if self._last_slot_ep_ids is not None:
            for ep_id in self._last_slot_ep_ids.tolist():
                if ep_id is not None and self._episode_meta.get(int(ep_id), {}).get("end_ts") is None:
                    self._episode_meta[int(ep_id)]["end_ts"] = now

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

        # Build the manifest up front with `written: false` for every episode.
        episodes_json: list[dict] = []
        for ep_id in sorted(episode_hits.keys()):
            hits = episode_hits[ep_id]
            meta = self._episode_meta.get(ep_id, {"slot": hits[0][1]})
            episodes_json.append({
                "episode_id": int(ep_id),
                "slot": int(meta.get("slot", hits[0][1])),
                "start_ts": meta.get("start_ts"),
                "end_ts": meta.get("end_ts"),
                "n_calls": meta.get("n_calls", len(hits)),
                "npz": f"episode_{ep_id:04d}.npz",
                "written": False,
            })

        def _write_manifest() -> None:
            """Atomic write of episodes.json (temp file + os.replace)."""
            tmp = self.record_attention_dir / "episodes.json.tmp"
            tmp.write_text(json.dumps(episodes_json, indent=2))
            tmp.replace(self.record_attention_dir / "episodes.json")

        _write_manifest()

        # Run-level config lands early too so `meta.json` is always present.
        meta_payload = {
            "n_calls_total": len(self._chunks),
            "n_episodes": len(episodes_json),
            "run_start": self._run_start_ts,
            "run_end": time.time(),
            "grip_max_left": self.grip_max_left,
            "grip_max_right": self.grip_max_right,
            "empty_hand_proprio": self.empty_hand_proprio,
            "language_override": self.language_override,
            "episode_source": "client_slot_ep_ids",
        }
        (self.record_attention_dir / "meta.json").write_text(
            json.dumps(meta_payload, indent=2)
        )

        # SIGINT tolerance: first Ctrl-C is buffered; second one raises.
        # We restore the original handler at the end.
        _saved_handler = signal.getsignal(signal.SIGINT)
        _interrupt_state = {"count": 0}

        def _on_sigint(signum, frame):
            _interrupt_state["count"] += 1
            if _interrupt_state["count"] == 1:
                print(
                    "\n[wrapper] Ctrl-C received during save — finishing current "
                    "episode and stopping. Press Ctrl-C again to abort immediately."
                )
            else:
                print("\n[wrapper] second Ctrl-C — aborting save.")
                signal.signal(signal.SIGINT, _saved_handler)
                raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except (ValueError, OSError):
            # signal.signal only works on the main thread; if we're being
            # called from an atexit chain in a non-main thread, just skip
            # the guard.
            pass

        try:
            for ep_index, ep in enumerate(episodes_json):
                if _interrupt_state["count"] >= 1:
                    print(f"[wrapper] stopping early — {ep_index}/{len(episodes_json)} episodes written")
                    break

                ep_id = ep["episode_id"]
                hits = episode_hits[ep_id]
                slot = ep["slot"]

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
                            per_call_slot.append(v.select(dim=slot_axis, index=s))
                        return torch.stack(per_call_slot, dim=0).numpy()
                    except Exception as e:
                        print(f"[wrapper] failed to slice '{key}' for ep {ep_id}: {e}")
                        return None

                cross = _slice_or_none("cross", slot_axis=2)
                if cross is not None:
                    payload["attentions"] = cross
                    payload["block_indices"] = cross_block_indices

                self_att = _slice_or_none("self", slot_axis=2)
                if self_att is not None:
                    payload["self_attentions"] = self_att
                    payload["self_block_indices"] = self_block_indices

                hidden = _slice_or_none("hidden", slot_axis=2)
                if hidden is not None:
                    payload["hidden_states"] = hidden
                    payload["hidden_block_indices"] = hidden_block_indices

                if image_mask is not None:
                    payload["image_mask"] = image_mask

                fname = ep["npz"]
                out_path = self.record_attention_dir / fname
                # Write to a `.tmp.npz` sibling and rename atomically so a
                # partially-written file never lingers.  Note the extension
                # order: np.savez appends `.npz` if the filename doesn't
                # already end in it — so we need `.tmp.npz`, not `.npz.tmp`,
                # or we'd end up writing `<name>.npz.tmp.npz`.
                tmp_path = self.record_attention_dir / (fname[:-4] + ".tmp.npz")
                print(
                    f"[wrapper] writing episode {ep_id:04d} "
                    f"({ep_index + 1}/{len(episodes_json)}, {len(hits)} calls) -> {fname}",
                    flush=True,
                )
                np.savez(tmp_path, **payload)
                tmp_path.replace(out_path)

                # Mark this episode written and refresh the manifest so a
                # crash between here and the next episode still leaves an
                # accurate index on disk.
                ep["written"] = True
                _write_manifest()
        finally:
            try:
                signal.signal(signal.SIGINT, _saved_handler)
            except (ValueError, OSError):
                pass

        # Also detach so the model is left clean if the process keeps living
        # for any reason (e.g. an atexit chain).
        try:
            if self.capture_handle is not None:
                self.capture_handle.detach()
        except Exception as e:
            print(f"[wrapper] capture_handle.detach failed: {e}")

        n_written = sum(1 for ep in episodes_json if ep.get("written"))
        print(
            f"[wrapper] saved {n_written}/{len(episodes_json)} per-episode npz files "
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
