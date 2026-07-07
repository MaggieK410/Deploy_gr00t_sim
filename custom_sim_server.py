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
        # DiT once, then reset()+read() per inference call. On shutdown, stack
        # everything into one attention.npz next to a meta.json.
        self.capture_handle = None
        self.record_attention_dir: Path | None = None
        self._chunks: list[dict] = []
        self._call_idx: int = 0
        self._run_start_ts: float = time.time()

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

        # Inference — bracket with attention capture reset/read.
        if self.capture_handle is not None:
            self.capture_handle.reset()

        action, info = self.policy.get_action(nested, options)

        if self.capture_handle is not None:
            captured = self.capture_handle.read()
            self._record_chunk(nested["video"][MODEL_VIDEO_KEY], captured)

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
    def _record_chunk(self, video_batch: np.ndarray, captured: dict) -> None:
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
        })
        self._call_idx += 1

    def _save_attention(self) -> None:
        """Flush self._chunks to attention.npz. Called at shutdown."""
        if not self._chunks or self.record_attention_dir is None:
            return
        out_path = self.record_attention_dir / "attention.npz"
        payload: dict[str, Any] = {}

        # Per-call metadata + input images.
        payload["call_idx"] = np.array([c["call_idx"] for c in self._chunks], dtype=np.int32)
        payload["timestamps"] = np.array([c["timestamp"] for c in self._chunks], dtype=np.float64)
        # Images: (N_calls, B, H, W, 3) uint8.
        payload["images"] = np.stack([c["image"] for c in self._chunks], axis=0)

        # Attention tensors: only stack if EVERY call produced them.
        def _stack_or_skip(key: str):
            vals = [c[key] for c in self._chunks]
            if all(v is not None for v in vals):
                try:
                    return torch.stack(vals, dim=0).numpy()
                except Exception as e:
                    print(f"[wrapper] failed to stack '{key}': {e}")
            return None

        cross = _stack_or_skip("cross")
        if cross is not None:
            payload["attentions"] = cross           # (N, n_denoise, n_cross_blocks, B, H, T_q, T_k_vlm)
            payload["block_indices"] = np.array(
                self._chunks[0]["cross_block_indices"], dtype=np.int32)

        self_att = _stack_or_skip("self")
        if self_att is not None:
            payload["self_attentions"] = self_att   # (N, n_denoise, n_self_blocks, B, H, T_q, T_q)
            payload["self_block_indices"] = np.array(
                self._chunks[0]["self_block_indices"], dtype=np.int32)

        hidden = _stack_or_skip("hidden")
        if hidden is not None:
            payload["hidden_states"] = hidden       # (N, n_denoise, n_blocks+1, B, T_q, D)
            payload["hidden_block_indices"] = np.array(
                self._chunks[0]["all_block_indices"], dtype=np.int32)

        # VLM image-vs-text mask (constant per run).
        if self.capture_handle is not None and self.capture_handle.image_mask is not None:
            payload["image_mask"] = self.capture_handle.image_mask.numpy().astype(np.bool_)

        # NOTE: savez (uncompressed) — same reasoning as deploy_groot.py:
        # compression is a 20-45s CPU wall for ~10% saving on random floats.
        np.savez(out_path, **payload)

        meta = {
            "n_calls": len(self._chunks),
            "run_start": self._run_start_ts,
            "run_end": time.time(),
            "grip_max_left": self.grip_max_left,
            "grip_max_right": self.grip_max_right,
            "empty_hand_proprio": self.empty_hand_proprio,
            "language_override": self.language_override,
            "keys_written": sorted(payload.keys()),
            "call_shapes": {
                k: list(v.shape) for k, v in payload.items()
                if hasattr(v, "shape")
            },
        }
        import json
        (self.record_attention_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Also detach so the model is left clean if the process keeps living
        # for any reason (e.g. an atexit chain).
        try:
            if self.capture_handle is not None:
                self.capture_handle.detach()
        except Exception as e:
            print(f"[wrapper] capture_handle.detach failed: {e}")

        print(f"[wrapper] saved attention capture: {out_path} ({len(self._chunks)} calls)")

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
