#!/usr/bin/env bash
# patch_robocasa_video_key_alias.sh
# ------------------------------------------------------------------
# Fixes the modality-key mismatch:
#
#   RuntimeError: Server error: Video key 'video.ego_view' must be in observation
#
# Root cause: robocasa-gr1-tabletop-tasks (at the pinned commit) emits
# the camera image under the LEGACY key `video.ego_view_pad_res256_freq20`
# (see `robocasa/models/robots/__init__.py:221` and
# `robocasa/utils/gym_utils/gymnasium_groot.py:31`). Modern GR00T
# checkpoints — including ones trained via the current data pipeline —
# expect the SHORT key `video.ego_view`.
#
# Fix: patch `gymnasium_groot.py` to add `video.ego_view` as an ALIAS of
# `video.ego_view_pad_res256_freq20`, both in the observation_space
# declaration and in the obs dict returned by `get_groot_observation`.
# Same underlying image; just an additional key.
#
# Usage
# -----
#   # Default: auto-detect via importable module
#   bash patch_robocasa_video_key_alias.sh
#
#   # Or point at a specific python that has robocasa installed
#   ROBOCASA_PYTHON=/path/to/venv/bin/python bash patch_robocasa_video_key_alias.sh
#
# Idempotent (marker check). Reversible (.bak).

set -euo pipefail

PY="${ROBOCASA_PYTHON:-python3}"

# Some robocasa imports print WARNING / info lines to stdout during
# import (e.g. "WARNING: mimicgen environments not imported ..."). We
# tag the real path with a distinctive prefix, then grep it out.
FILE=$($PY - <<'PY' 2>/dev/null | sed -n 's/^__PATCH_TARGET__=//p' | tail -n 1
import importlib.util
spec = importlib.util.find_spec("robocasa.utils.gym_utils.gymnasium_groot")
if spec is None or spec.origin is None:
    raise SystemExit("robocasa.utils.gym_utils.gymnasium_groot not importable — "
                     "activate the correct venv first.")
print(f"__PATCH_TARGET__={spec.origin}")
PY
)

if [ -z "$FILE" ]; then
    echo "ERROR: could not locate gymnasium_groot.py — is robocasa importable in this venv?"
    exit 1
fi

echo "[patch] target: $FILE"

if grep -q "PATCHED_VIDEO_EGO_VIEW_ALIAS" "$FILE"; then
    echo "[patch] $FILE already patched — no-op."
    grep -n "PATCHED_VIDEO_EGO_VIEW_ALIAS" "$FILE" | head -3
    exit 0
fi

echo "[patch] backing up $FILE -> $FILE.bak"
cp "$FILE" "$FILE.bak"

$PY - "$FILE" <<'PY'
"""Two injections in `gymnasium_groot.py`:

(A) In `GrootRoboCasaEnv.__init__`, right after the `if mapped_name ==
    "video.ego_view_pad_res256_freq20":` block that adds the co-train
    key to observation_space, also add `video.ego_view` as another
    alias to the same space.

(B) In `get_groot_observation`, right after the corresponding block that
    populates `video.ego_view_bg_crop_pad_res256_freq20`, also copy
    the processed image into `video.ego_view`.

Both insertions preserve the captured indent of the anchor line.
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()

MARKER = "PATCHED_VIDEO_EGO_VIEW_ALIAS"

# Anchor 1 (in __init__): the block that adds bg_crop to observation_space.
# We inject a second alias right after that closing block.
anchor_init = re.compile(
    r"(?P<indent>[ \t]+)if mapped_name == \"video\.ego_view_pad_res256_freq20\":\s*\n"
    r"(?:[ \t]+.*\n)+?"                                # non-greedy body
    r"[ \t]+\)\s*\n"                                   # closing paren of spaces.Box(...)
)
# Anchor 2 (in get_groot_observation): the block that assigns
# obs["video.ego_view_bg_crop_pad_res256_freq20"] = process_img_cotrain(...)
anchor_step = re.compile(
    r"(?P<indent>[ \t]+)if mapped_name == \"video\.ego_view_pad_res256_freq20\":\s*\n"
    r"(?:[ \t]+.*\n)+?"                                # non-greedy body (the obs[...] = ... call)
    r"[ \t]+\)\s*\n"                                   # closing paren of process_img_cotrain(...)
)
# Both anchors have identical STRUCTURE (both use the same `if` line),
# so we distinguish by finding them in order — first occurrence is init,
# second is get_groot_observation.

matches = list(anchor_init.finditer(src))
if len(matches) < 2:
    sys.exit(
        f"PATCH FAILED: expected 2 anchor blocks in {path}, "
        f"found {len(matches)}. Edit by hand."
    )

# Injection for anchor 1 (observation_space).
def make_space_alias(indent: str) -> str:
    return (
        f"{indent}# {MARKER} — expose short-name key for newer checkpoints\n"
        f"{indent}self.observation_space[\"video.ego_view\"] = spaces.Box(\n"
        f"{indent}    low=0, high=255, shape=(*FINAL_IMAGE_RESOLUTION, 3), dtype=np.uint8\n"
        f"{indent})\n"
    )

# Injection for anchor 2 (obs dict).
def make_obs_alias(indent: str) -> str:
    return (
        f"{indent}# {MARKER} — mirror image under short key\n"
        f"{indent}obs[\"video.ego_view\"] = obs[\"video.ego_view_pad_res256_freq20\"]\n"
    )

# Insert in REVERSE order so earlier offsets aren't shifted.
m2 = matches[1]
m1 = matches[0]

indent2 = m2.group("indent")
inj2 = make_obs_alias(indent2)
src = src[: m2.end()] + inj2 + src[m2.end():]

indent1 = m1.group("indent")
inj1 = make_space_alias(indent1)
src = src[: m1.end()] + inj1 + src[m1.end():]

open(path, "w").write(src)
print(f"[patch] Patched {path}")
PY

echo "[patch] verification:"
grep -n "PATCHED_VIDEO_EGO_VIEW_ALIAS\|video.ego_view" "$FILE" | head -12

echo ""
echo "[patch] Sanity import check:"
$PY - <<'PY'
import robocasa.utils.gym_utils.gymnasium_groot as g
src = open(g.__file__).read()
assert 'obs["video.ego_view"] = obs["video.ego_view_pad_res256_freq20"]' in src, \
    "PATCH DID NOT LAND — check the file by hand."
assert 'self.observation_space["video.ego_view"] = spaces.Box(' in src, \
    "PATCH DID NOT LAND (space) — check the file by hand."
print("  ✓ both aliases present in", g.__file__)
PY

echo ""
echo "[patch] Done. Re-run the client — the 'video.ego_view' error should be gone."
echo "        (State keys and language key already match. Next error, if any,"
echo "         will be about action-key naming or shape.)"
echo ""
echo "To revert:  mv $FILE.bak $FILE"
