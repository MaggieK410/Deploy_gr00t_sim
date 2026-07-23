#!/usr/bin/env python3
"""
pair_videos_to_episodes.py
--------------------------
After a rollout completes, this walks the video output directory from
`rollout_policy.py` (default `/tmp/sim_eval_videos_*_ac*/`) and the
attention output directory from `custom_sim_server.py` (an
`episodes.json` under `<record-attention-dir>/run_<timestamp>/`), and
copies each mp4 into `<record-attention-dir>/run_<timestamp>/videos/`
renamed to `episode_XXXX_sN.mp4` so it lines up 1:1 with the matching
`episode_XXXX.npz`.

Pairing method
--------------
The rollout wrapper writes mp4s with UUID names, then RENAMES each to
`<uuid>_s{success}[_g-o…].mp4` at episode reset.  Both the server and
the wrapper stamp their "episode ends" in roughly the same order (the
async-vec-env slot that finishes first triggers a reset first).  So we
sort mp4s by mtime and episodes.json entries by `end_ts`, and zip them.

If you have more mp4s than episodes (or vice-versa — e.g. the final
open episode never got a reset event on one of the sides) the extras
are left unmatched and reported.

Usage
-----
    python pair_videos_to_episodes.py \\
        --attention-dir  ~/sim_attention_runs/run_20260707_175918/ \\
        [--video-dir     /tmp/sim_eval_videos_*_ac16*/]        \\
        [--copy | --link | --move]

`--video-dir` may be a glob.  If not given, the newest matching
`/tmp/sim_eval_videos_*_ac*` directory is used.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path


def _resolve_video_dir(spec: str | None) -> Path:
    if spec is None:
        candidates = sorted(
            glob.glob("/tmp/sim_eval_videos_*_ac*"),
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit(
                "No /tmp/sim_eval_videos_*_ac* directories found. "
                "Pass --video-dir explicitly."
            )
        return Path(candidates[0])
    matches = glob.glob(spec)
    if len(matches) != 1:
        raise SystemExit(
            f"--video-dir glob '{spec}' matched {len(matches)} paths; "
            f"want exactly 1. Matches: {matches}"
        )
    return Path(matches[0])


def _load_episodes(attn_dir: Path) -> list[dict]:
    """Load episodes.json if it exists.  If not, reconstruct a best-effort
    manifest by scanning any `episode_XXXX.npz` files in the run directory.

    The reconstruction reads each npz's `episode_id`, `slot`, and the first +
    last `timestamps` value to fill in `start_ts` / `end_ts` — enough for the
    mtime-based video pairing below to work.
    """
    ep_path = attn_dir / "episodes.json"
    if ep_path.exists():
        return json.loads(ep_path.read_text())

    print(f"[pair] {ep_path.name} missing — reconstructing from .npz files")
    import numpy as np  # local import so the script is fast when the manifest exists
    npzs = sorted(attn_dir.glob("episode_*.npz"))
    if not npzs:
        raise SystemExit(
            f"neither {ep_path} nor any episode_*.npz files under {attn_dir}. "
            f"Did the server run at all with --record-attention-dir?"
        )

    reconstructed: list[dict] = []
    for p in npzs:
        try:
            with np.load(p, allow_pickle=False) as z:
                ep_id = int(z["episode_id"])
                slot = int(z["slot"])
                ts = z["timestamps"]
                start_ts = float(ts[0]) if len(ts) else None
                end_ts = float(ts[-1]) if len(ts) else None
                n_calls = int(len(ts))
        except Exception as e:
            print(f"[pair]   skipping {p.name}: {e}")
            continue
        reconstructed.append({
            "episode_id": ep_id,
            "slot": slot,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "n_calls": n_calls,
            "npz": p.name,
            "written": True,
        })
    reconstructed.sort(key=lambda e: e["episode_id"])
    print(f"[pair]   reconstructed {len(reconstructed)} episodes from .npz files")
    return reconstructed


def _list_videos_by_mtime(video_dir: Path) -> list[Path]:
    mp4s = sorted(video_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return mp4s


def _final_success_suffix(mp4_name: str) -> str:
    """Extract the `_s0`/`_s1` (+ any g-o / not-g-d) suffix the video wrapper
    appended at reset time, so we can carry it through the rename."""
    # video wrapper produces names like `<uuid>_s1_g-o1_not-g-d1.mp4`.
    stem = Path(mp4_name).stem
    tail_start = stem.find("_s")
    if tail_start == -1:
        return ""
    return stem[tail_start:]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attention-dir", type=Path, required=True,
                   help="the run_<timestamp>/ folder that custom_sim_server.py wrote")
    p.add_argument("--video-dir", type=str, default=None,
                   help="source dir of mp4s from rollout_policy.py "
                        "(default: newest /tmp/sim_eval_videos_*_ac*)")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--copy", dest="mode", action="store_const", const="copy",
                        default="copy", help="copy files (default)")
    action.add_argument("--link", dest="mode", action="store_const", const="link",
                        help="hard-link instead of copying (fast, saves disk)")
    action.add_argument("--move", dest="mode", action="store_const", const="move",
                        help="move (rm the source)")
    args = p.parse_args(argv)

    attn_dir = args.attention_dir.expanduser().resolve()
    video_dir = _resolve_video_dir(args.video_dir).resolve()
    print(f"[pair] attention run:  {attn_dir}")
    print(f"[pair] video source:   {video_dir}")

    episodes = _load_episodes(attn_dir)
    mp4s = _list_videos_by_mtime(video_dir)
    print(f"[pair] {len(episodes)} episodes in episodes.json, {len(mp4s)} mp4s on disk")

    # Sort episodes by end_ts (matches mp4 mtime order the wrapper renames on
    # reset). Episodes with no end_ts (still open at shutdown) go last.
    episodes_by_end = sorted(
        episodes,
        key=lambda e: (e.get("end_ts") is None, e.get("end_ts") or 0.0),
    )

    out_dir = attn_dir / "videos"
    out_dir.mkdir(exist_ok=True)

    n_paired = 0
    for ep, mp4 in zip(episodes_by_end, mp4s):
        ep_id = ep["episode_id"]
        suffix = _final_success_suffix(mp4.name) or "_s?"
        target = out_dir / f"episode_{ep_id:04d}{suffix}.mp4"
        if args.mode == "copy":
            shutil.copy2(mp4, target)
        elif args.mode == "link":
            if target.exists():
                target.unlink()
            target.hardlink_to(mp4)
        else:
            shutil.move(str(mp4), str(target))
        print(f"[pair] episode {ep_id:04d} <- {mp4.name}  ->  {target.name}")
        n_paired += 1

    # Report the unpaired remainder (either side).
    if len(mp4s) > n_paired:
        print(f"[pair] WARNING: {len(mp4s) - n_paired} mp4s left unpaired "
              f"(more videos than episodes in episodes.json)")
    if len(episodes_by_end) > n_paired:
        print(f"[pair] WARNING: {len(episodes_by_end) - n_paired} episodes left without a video "
              f"(more episodes than mp4s on disk)")

    print(f"[pair] done. {n_paired} pairs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
