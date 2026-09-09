"""Custom GR00T inference server for base / unmodified GR00T checkpoints.

Sibling of ``custom_sim_server.py``:
  * ``custom_sim_server.py`` bakes in the finetune-specific action schema
    (per-part env action keys, planner-order upper split, grip synthesis).
    Only works for models finetuned on the merged G1 sim dataset.
  * THIS file wraps ``Gr00tSimPolicyWrapper`` — GR00T's stock rollout
    wrapper — so the obs/action pipeline matches whatever the loaded model
    was trained on, without any per-part hardcoding.  Use this for base
    GR00T checkpoints and for any finetune whose action schema differs
    from the merged G1 pipeline.

What we keep from ``custom_sim_server.py``:
  * Streaming per-episode attention + hidden-state capture (same npz
    schema, so ``pair_videos_to_episodes.py`` continues to work).
  * ``slot_ep_ids``-driven per-episode bookkeeping (from the rollout
    patch).
  * SIGINT-tolerant shutdown flush.

What we drop:
  * Per-part action reassembly and PLANNER_FROM_LOGICAL split.
  * Grip synthesis + grip_max_* knobs.
  * Language override / prefix stripping (`Gr00tSimPolicyWrapper` does
    obs pass-through as-is; the model reads the language field the env
    already provides).

Usage
-----
    python custom_sim_server_gr00t_base.py \\
        --model-path /path/to/gr00t/base/checkpoint \\
        --embodiment-tag GR1 --device cuda:0 --port 5555 \\
        --record-attention-dir ./attention_out_gr00t_base
"""

import atexit
from dataclasses import dataclass
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer


# ══════════════════════════════════════════════════════════════════════════════
# Wrapper — extends Gr00tSimPolicyWrapper with attention capture
# ══════════════════════════════════════════════════════════════════════════════
class Gr00tBaseSimWrapper(Gr00tSimPolicyWrapper):
    """``Gr00tSimPolicyWrapper`` + the streaming per-episode attention capture
    from ``custom_sim_server.CustomSimWrapper``.

    The action pipeline is entirely inherited from ``Gr00tSimPolicyWrapper`` —
    whatever obs the env produces gets passed to the underlying ``Gr00tPolicy``,
    and whatever action the model outputs gets passed back to the env.  We only
    add the capture bracketing around the ``super()._get_action`` call.

    Everything below the ``─── attention capture ───`` divider is copied
    from ``custom_sim_server.py`` verbatim, so the on-disk output schema and
    manifest format match exactly.
    """

    def __init__(
        self,
        policy: Gr00tPolicy,
        *,
        strict: bool = True,
        record_attention_dir: str | None = None,
    ):
        super().__init__(policy, strict=strict)
        # store the concrete policy so we can reach into its DiT for capture
        self.policy = policy

        # One-shot shape dumps (analogous to custom_sim_server.py).
        self._dumped_obs_shapes = False
        self._dumped_action_shapes = False

        # ── Attention capture bookkeeping (same structure as custom_sim_server.py) ──
        self.capture_handle = None
        self.record_attention_dir: Path | None = None
        self._open_episodes: dict[int, list[dict]] = {}
        self._closed_episodes: set[int] = set()
        self._episodes_json: list[dict] = []
        self._episodes_json_by_id: dict[int, dict] = {}
        self._meta_written: bool = False
        self._cross_block_indices: np.ndarray | None = None
        self._self_block_indices: np.ndarray | None = None
        self._hidden_block_indices: np.ndarray | None = None
        self._n_calls_total: int = 0
        self._call_idx: int = 0
        self._run_start_ts: float = time.time()
        self._last_slot_ep_ids: np.ndarray | None = None
        self._episode_meta: dict[int, dict] = {}
        self._warned_missing_ep_ids: bool = False

        # Introspect modality configs for sanity — same as custom_sim_server.py
        try:
            state_cfg = policy.modality_configs["state"]
            action_cfg = policy.modality_configs["action"]
            video_cfg = policy.modality_configs["video"]
            print(f"[wrapper] model state modality_keys : {list(state_cfg.modality_keys)}")
            print(f"[wrapper] model action modality_keys: {list(action_cfg.modality_keys)}")
            print(f"[wrapper] model video modality_keys : {list(video_cfg.modality_keys)}")
        except Exception as e:
            print(f"[wrapper] could not introspect model modality configs: {e}")

        if record_attention_dir is not None:
            self._init_attention_capture(policy, record_attention_dir)

    def _init_attention_capture(self, policy: Gr00tPolicy, out_dir: str) -> None:
        try:
            from capture_attention import attach as _attach_capture
        except ImportError as e:
            raise RuntimeError(
                "record_attention_dir set but capture_attention.py isn't importable. "
                "Copy Deploy_gr00t_sim/capture_attention.py to a directory on PYTHONPATH."
            ) from e
        try:
            dit = policy.model.action_head.model
        except AttributeError as e:
            raise RuntimeError(
                "Could not find policy.model.action_head.model — expected the DiT "
                "(or AlternateVLDiT). Check the checkpoint's model layout."
            ) from e
        self.capture_handle = _attach_capture(dit)

        run_id = time.strftime("%Y%m%d_%H%M%S")
        self.record_attention_dir = Path(out_dir) / f"run_{run_id}"
        self.record_attention_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self._save_attention)

        print(
            f"[wrapper] attention capture attached — {len(self.capture_handle.cross_block_indices)} "
            f"cross blocks, {len(self.capture_handle.self_block_indices)} self blocks; "
            f"writing to {self.record_attention_dir}"
        )

    # ══════════════════════════════════════════════════════════════════
    # Main action pipeline — delegate to Gr00tSimPolicyWrapper, bracket
    # with attention capture reset / read + per-episode bookkeeping
    # ══════════════════════════════════════════════════════════════════
    def _get_action(self, observation: dict[str, Any], options=None):
        # Per-slot episode ids from the patched rollout_policy.py.
        slot_ep_id_this_call = self._read_slot_ep_ids(observation)

        if not self._dumped_obs_shapes:
            self._dump_obs_shapes(observation)
            self._dumped_obs_shapes = True

        # Reset capture, delegate, read.
        if self.capture_handle is not None:
            self.capture_handle.reset()

        # Everything the stock wrapper does — obs remap, model call, action
        # remap — happens inside super()._get_action.  We don't touch it.
        action, info = super()._get_action(observation, options)

        if self.capture_handle is not None:
            captured = self.capture_handle.read()
            # Video for the images npz — pull whichever key(s) the env
            # exposes.  Fall back to the first video key we find so we
            # always have something to save.
            video_batch = self._extract_video_batch(observation)
            self._record_chunk(
                video_batch=video_batch,
                captured=captured,
                slot_ep_ids=slot_ep_id_this_call,
            )

        if not self._dumped_action_shapes:
            print("[wrapper] --- first action: shape dump ---")
            for k, v in action.items():
                arr = np.asarray(v)
                print(f"[wrapper]   {k}: shape={arr.shape}, dtype={arr.dtype}")
            print("[wrapper] --- end action dump ---")
            self._dumped_action_shapes = True

        return action, info

    def _extract_video_batch(self, observation: dict[str, Any]) -> np.ndarray | None:
        """Pull a video tensor for the images npz.  Prefers the ego view key
        the env normally emits; falls back to the first video.* key we find."""
        preferred = (
            "video.ego_view_pad_res256_freq20",
            "video.ego_view",
        )
        for k in preferred:
            if k in observation:
                return np.asarray(observation[k])
        # Fall back to any video.* key.
        for k in observation.keys():
            if isinstance(k, str) and k.startswith("video."):
                return np.asarray(observation[k])
        return None

    def _dump_obs_shapes(self, observation: dict[str, Any]) -> None:
        print("[wrapper] --- first obs: shape dump ---")
        for k, v in observation.items():
            try:
                arr = np.asarray(v)
                print(f"[wrapper]   {k}: shape={arr.shape}, dtype={arr.dtype}")
            except Exception:
                print(f"[wrapper]   {k}: type={type(v).__name__}, value={v!r}")
        print("[wrapper] --- end obs dump ---")

    # ══════════════════════════════════════════════════════════════════
    # ─── attention capture ───
    # Copied verbatim from custom_sim_server.CustomSimWrapper so the
    # on-disk format (episodes.json, episode_XXXX.npz, meta.json) is
    # byte-for-byte compatible with pair_videos_to_episodes.py.
    # ══════════════════════════════════════════════════════════════════
    def _infer_batch_size(self, observation: dict[str, Any]) -> int:
        v = observation.get("video.ego_view_pad_res256_freq20")
        if v is None:
            # Try alternates or fall back to state.
            for k in observation.keys():
                if isinstance(k, str) and k.startswith("video."):
                    v = observation[k]
                    break
        if v is not None:
            return len(v)
        # If no video, fall back to any state.*.
        for k in observation.keys():
            if isinstance(k, str) and k.startswith("state."):
                return len(np.asarray(observation[k]))
        return 1

    def _read_slot_ep_ids(self, observation: dict[str, Any]) -> np.ndarray:
        batch_size = self._infer_batch_size(observation)
        raw = observation.get("slot_ep_ids")
        if raw is None:
            if not self._warned_missing_ep_ids:
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

        def _is_valid(ep_id: int) -> bool:
            return ep_id >= 0

        if self._last_slot_ep_ids is None:
            self._last_slot_ep_ids = slot_ep_ids.copy()
            for slot, ep_id in enumerate(slot_ep_ids.tolist()):
                if not _is_valid(ep_id):
                    continue
                self._episode_meta.setdefault(int(ep_id), {
                    "episode_id": int(ep_id),
                    "slot": int(slot),
                    "start_ts": now,
                    "end_ts": None,
                    "n_calls": 0,
                })
                self._add_manifest_entry(int(ep_id), int(slot), now)
            if self.record_attention_dir is not None and self._episodes_json:
                self._write_manifest()
            transitioned_out: list[int] = []
        else:
            transitioned_out = []
            for slot in range(batch_size):
                incoming = int(slot_ep_ids[slot])
                outgoing = int(self._last_slot_ep_ids[slot])
                if incoming != outgoing:
                    if _is_valid(outgoing) and outgoing in self._episode_meta \
                            and self._episode_meta[outgoing]["end_ts"] is None:
                        self._episode_meta[outgoing]["end_ts"] = now
                        transitioned_out.append(outgoing)
                    if _is_valid(incoming):
                        self._episode_meta.setdefault(incoming, {
                            "episode_id": incoming,
                            "slot": int(slot),
                            "start_ts": now,
                            "end_ts": None,
                            "n_calls": 0,
                        })
                        self._add_manifest_entry(incoming, int(slot), now)
            self._last_slot_ep_ids = slot_ep_ids.copy()

        for slot in range(batch_size):
            ep_id = int(slot_ep_ids[slot])
            if _is_valid(ep_id) and ep_id in self._episode_meta:
                self._episode_meta[ep_id]["n_calls"] += 1

        for ep_id in transitioned_out:
            self._flush_episode(ep_id)

        return slot_ep_ids

    @staticmethod
    def _slot_slice_np(t, slot: int):
        if t is None:
            return None
        return t.select(dim=2, index=slot).contiguous().cpu().numpy()

    def _record_chunk(
        self,
        video_batch,
        captured: dict,
        slot_ep_ids: np.ndarray,
    ) -> None:
        img = None
        if video_batch is not None:
            img = np.asarray(video_batch)
            if img.ndim == 5:
                img = img[:, -1]
            img = np.ascontiguousarray(img, dtype=np.uint8)

        call_idx = self._call_idx
        self._call_idx += 1
        self._n_calls_total += 1
        now = time.time() - self._run_start_ts

        cross_full = captured.get("cross")
        self_full = captured.get("self")
        hidden_full = captured.get("hidden_states")

        if self._cross_block_indices is None:
            self._cross_block_indices = np.array(captured.get("cross_block_indices", []), dtype=np.int32)
            self._self_block_indices = np.array(captured.get("self_block_indices", []), dtype=np.int32)
            self._hidden_block_indices = np.array(captured.get("all_block_indices", []), dtype=np.int32)

        for slot, ep_id in enumerate(slot_ep_ids.tolist()):
            ep_id = int(ep_id)
            if ep_id < 0 or ep_id in self._closed_episodes:
                continue
            entry = {
                "call_idx": call_idx,
                "timestamp": now,
                "cross":  self._slot_slice_np(cross_full,  slot),
                "self":   self._slot_slice_np(self_full,   slot),
                "hidden": self._slot_slice_np(hidden_full, slot),
            }
            if img is not None and slot < len(img):
                entry["image"] = img[slot].copy()
            self._open_episodes.setdefault(ep_id, []).append(entry)

    def _add_manifest_entry(self, ep_id: int, slot: int, start_ts: float) -> dict:
        if ep_id in self._episodes_json_by_id:
            return self._episodes_json_by_id[ep_id]
        entry = {
            "episode_id": int(ep_id),
            "slot": int(slot),
            "start_ts": float(start_ts),
            "end_ts": None,
            "n_calls": 0,
            "npz": f"episode_{ep_id:04d}.npz",
            "written": False,
        }
        self._episodes_json.append(entry)
        self._episodes_json_by_id[ep_id] = entry
        return entry

    def _write_manifest(self) -> None:
        if self.record_attention_dir is None:
            return
        import json
        for entry in self._episodes_json:
            meta = self._episode_meta.get(entry["episode_id"], {})
            if "n_calls" in meta:
                entry["n_calls"] = int(meta["n_calls"])
            if meta.get("end_ts") is not None:
                entry["end_ts"] = float(meta["end_ts"])
        tmp = self.record_attention_dir / "episodes.json.tmp"
        tmp.write_text(json.dumps(self._episodes_json, indent=2))
        tmp.replace(self.record_attention_dir / "episodes.json")

    def _maybe_write_meta(self) -> None:
        if self._meta_written or self.record_attention_dir is None:
            return
        import json
        payload = {
            "run_start": self._run_start_ts,
            "server_variant": "gr00t_base",
            "episode_source": "client_slot_ep_ids",
            "save_mode": "streaming_per_episode",
        }
        (self.record_attention_dir / "meta.json").write_text(
            json.dumps(payload, indent=2)
        )
        self._meta_written = True

    def _flush_episode(self, ep_id: int) -> None:
        if self.record_attention_dir is None:
            return
        if ep_id in self._closed_episodes:
            return
        chunks = self._open_episodes.get(ep_id)
        if not chunks:
            self._closed_episodes.add(ep_id)
            return

        self._maybe_write_meta()

        entry = self._episodes_json_by_id.get(ep_id)
        if entry is None:
            entry = self._add_manifest_entry(ep_id, 0, chunks[0]["timestamp"])

        call_indices = np.array([c["call_idx"]  for c in chunks], dtype=np.int32)
        timestamps   = np.array([c["timestamp"] for c in chunks], dtype=np.float64)

        payload: dict[str, Any] = {
            "episode_id": np.int32(ep_id),
            "slot":       np.int32(entry["slot"]),
            "call_idx":   call_indices,
            "timestamps": timestamps,
        }

        images = [c.get("image") for c in chunks]
        if all(im is not None for im in images):
            try:
                payload["images"] = np.stack(images, axis=0)
            except Exception as e:
                print(f"[wrapper] failed to stack images for ep {ep_id}: {e}")

        def _stack_or_none(key: str):
            vals = [c.get(key) for c in chunks]
            if any(v is None for v in vals):
                return None
            try:
                return np.stack(vals, axis=0)
            except Exception as e:
                print(f"[wrapper] failed to stack '{key}' for ep {ep_id}: {e}")
                return None

        cross = _stack_or_none("cross")
        if cross is not None:
            payload["attentions"] = cross
            payload["block_indices"] = self._cross_block_indices

        self_att = _stack_or_none("self")
        if self_att is not None:
            payload["self_attentions"] = self_att
            payload["self_block_indices"] = self._self_block_indices

        hidden = _stack_or_none("hidden")
        if hidden is not None:
            payload["hidden_states"] = hidden
            payload["hidden_block_indices"] = self._hidden_block_indices

        if self.capture_handle is not None and self.capture_handle.image_mask is not None:
            payload["image_mask"] = self.capture_handle.image_mask.numpy().astype(np.bool_)

        self._write_manifest()

        fname = entry["npz"]
        out_path = self.record_attention_dir / fname
        tmp_path = self.record_attention_dir / (fname[:-4] + ".tmp.npz")
        print(f"[wrapper] flushing episode {ep_id:04d} ({len(chunks)} calls) -> {fname}",
              flush=True)
        np.savez(tmp_path, **payload)
        tmp_path.replace(out_path)

        entry["written"] = True
        self._write_manifest()

        del self._open_episodes[ep_id]
        self._closed_episodes.add(ep_id)

    def _save_attention(self) -> None:
        if self.record_attention_dir is None or self.capture_handle is None:
            return
        if getattr(self, "_save_done", False):
            return
        self._save_done = True

        now = time.time() - self._run_start_ts
        if self._last_slot_ep_ids is not None:
            for ep_id in self._last_slot_ep_ids.tolist():
                ep_id = int(ep_id)
                if ep_id < 0 or ep_id in self._closed_episodes:
                    continue
                meta = self._episode_meta.get(ep_id)
                if meta is not None and meta.get("end_ts") is None:
                    meta["end_ts"] = now

        remaining = sorted(self._open_episodes.keys())
        if not remaining:
            print(
                f"[wrapper] shutdown: no open episodes to flush "
                f"({len(self._closed_episodes)} already streamed, "
                f"{self._n_calls_total} calls total) → {self.record_attention_dir}"
            )
            try:
                self.capture_handle.detach()
            except Exception as e:
                print(f"[wrapper] capture_handle.detach failed: {e}")
            return

        _saved_handler = signal.getsignal(signal.SIGINT)
        _interrupt_state = {"count": 0}

        def _on_sigint(signum, frame):
            _interrupt_state["count"] += 1
            if _interrupt_state["count"] == 1:
                print(
                    "\n[wrapper] Ctrl-C during shutdown flush — finishing "
                    "current episode and stopping. Ctrl-C again to abort."
                )
            else:
                print("\n[wrapper] second Ctrl-C — aborting remaining flushes.")
                signal.signal(signal.SIGINT, _saved_handler)
                raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except (ValueError, OSError):
            pass

        try:
            for i, ep_id in enumerate(remaining):
                if _interrupt_state["count"] >= 1:
                    print(f"[wrapper] stopping early — {i}/{len(remaining)} open episodes flushed")
                    break
                self._flush_episode(ep_id)
        finally:
            try:
                signal.signal(signal.SIGINT, _saved_handler)
            except (ValueError, OSError):
                pass

        try:
            self.capture_handle.detach()
        except Exception as e:
            print(f"[wrapper] capture_handle.detach failed: {e}")

        n_written = sum(1 for ep in self._episodes_json if ep.get("written"))
        print(
            f"[wrapper] shutdown: {n_written}/{len(self._episodes_json)} episodes saved "
            f"({self._n_calls_total} calls total) → {self.record_attention_dir}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ServerConfig:
    model_path: str
    """Path to the model checkpoint directory."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.GR1
    """Embodiment tag for the loaded GR00T model.  Base GR00T checkpoints use
    GR1 (or one of the other public tags); NOT NEW_EMBODIMENT unless your
    checkpoint was finetuned with that tag."""

    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 5555

    strict: bool = True

    record_attention_dir: str | None = None
    """If set, attach a CaptureHandle to the DiT and stream per-episode
    attention + hidden states to <this>/run_YYYYMMDD_HHMMSS/.
    Requires capture_attention.py on PYTHONPATH."""


def main(cfg: ServerConfig):
    print(f"[server] loading {cfg.model_path} on {cfg.device}")
    policy = Gr00tPolicy(
        embodiment_tag=cfg.embodiment_tag,
        model_path=cfg.model_path,
        device=cfg.device,
        strict=cfg.strict,
    )
    wrapped = Gr00tBaseSimWrapper(
        policy,
        strict=cfg.strict,
        record_attention_dir=cfg.record_attention_dir,
    )
    print(f"[server] listening on {cfg.host}:{cfg.port}")
    server = PolicyServer(policy=wrapped, host=cfg.host, port=cfg.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[server] shutdown")
    finally:
        wrapped._save_attention()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
