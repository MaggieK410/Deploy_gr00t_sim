#!/usr/bin/env bash
# patch_robocasa_video_key_alias.sh
# ------------------------------------------------------------------
# Aligns robocasa-gr1-tabletop-tasks' modality key naming with modern
# GR00T checkpoints. The env at the pinned commit uses legacy names:
#
#   video.ego_view_pad_res256_freq20     (image)
#   annotation.human.coarse_action       (prefixed with "unlocked_waist: ")
#
# Modern checkpoints (finetunes on datasets exported with the newer
# pipeline) expect:
#
#   video.ego_view                       (short name)
#   annotation.human.action.task_description   (raw text, no prefix)
#
# This patch adds ALIAS keys — the underlying image/text is unchanged,
# just exposed under an additional name. Two independent modality
# aliases in a single script:
#
#   [A] video.ego_view_pad_res256_freq20  -> video.ego_view
#   [B] annotation.human.coarse_action    -> annotation.human.action.task_description
#       (with "locked_waist: " / "unlocked_waist: " prefix stripped)
#
# Both are per-marker idempotent; you can re-run this script as many
# times as you want and each alias only lands once.
#
# Usage
# -----
#   # Default: auto-detect via importable module
#   bash patch_robocasa_video_key_alias.sh
#
#   # Or point at a specific python that has robocasa installed
#   ROBOCASA_PYTHON=/path/to/venv/bin/python bash patch_robocasa_video_key_alias.sh
#
# Reversible (.bak preserved on first run only, so it always reflects
# the pristine pre-patch file).

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

# Preserve the pristine .bak on FIRST patch run; don't clobber it on re-runs.
if [ ! -f "$FILE.bak" ]; then
    echo "[patch] backing up $FILE -> $FILE.bak"
    cp "$FILE" "$FILE.bak"
else
    echo "[patch] $FILE.bak already exists — leaving it alone (represents pristine file)"
fi

$PY - "$FILE" <<'PY'
"""Four injections in `gymnasium_groot.py`, grouped into two independent
markers so you can re-run this script and only unpatched sections land.

VIDEO alias (marker: PATCHED_VIDEO_EGO_VIEW_ALIAS):
  A1: In `__init__`, after the `if mapped_name == "video.ego_view_pad_res256_freq20":`
      block that adds the co-train key to observation_space, also declare
      `video.ego_view` as a Box of the same shape.
  A2: In `get_groot_observation`, after the corresponding block that
      populates `video.ego_view_bg_crop_pad_res256_freq20`, copy the
      processed image into `video.ego_view`.

LANGUAGE alias (marker: PATCHED_LANGUAGE_TASK_DESC_ALIAS):
  B1: In `__init__`, right after the `elif isinstance(..., GR1ArmsAndWaist):`
      block that adds `annotation.human.coarse_action` to observation_space,
      also declare `annotation.human.action.task_description` as Text.
  B2: In `get_groot_observation`, right after the corresponding elif
      block that assigns `"unlocked_waist: {raw_obs['language']}"`, also
      assign the RAW text (prefix stripped) to
      `annotation.human.action.task_description`.

Each injection is guarded by an in-body marker check so re-runs are safe.
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()

VIDEO_MARKER = "PATCHED_VIDEO_EGO_VIEW_ALIAS"
LANG_MARKER = "PATCHED_LANGUAGE_TASK_DESC_ALIAS"

# ----- VIDEO alias -----
# Anchor structure (appears twice — first in __init__, second in
# get_groot_observation):
#     <indent>if mapped_name == "video.ego_view_pad_res256_freq20":
#     <indent>    <body>
#     <indent>    )
video_anchor = re.compile(
    r"(?P<indent>[ \t]+)if mapped_name == \"video\.ego_view_pad_res256_freq20\":\s*\n"
    r"(?:[ \t]+.*\n)+?"
    r"[ \t]+\)\s*\n"
)

def _video_space_injection(indent: str) -> str:
    return (
        f"{indent}# {VIDEO_MARKER} — expose short-name key for newer checkpoints\n"
        f"{indent}self.observation_space[\"video.ego_view\"] = spaces.Box(\n"
        f"{indent}    low=0, high=255, shape=(*FINAL_IMAGE_RESOLUTION, 3), dtype=np.uint8\n"
        f"{indent})\n"
    )

def _video_obs_injection(indent: str) -> str:
    return (
        f"{indent}# {VIDEO_MARKER} — mirror image under short key\n"
        f"{indent}obs[\"video.ego_view\"] = obs[\"video.ego_view_pad_res256_freq20\"]\n"
    )

# ----- LANGUAGE alias -----
# Anchor B1 (in __init__): the line RIGHT AFTER the whole
#   if/elif/else annotation.human.* block. This is uniquely identified
#   by `self.action_space = self.key_converter.deduce_action_space(self.env)`.
#   We insert BEFORE this line so our declaration is at the same
#   indentation as the surrounding statements — outside the if/elif/else.
lang_space_anchor = re.compile(
    r"(?P<indent>[ \t]+)self\.action_space = self\.key_converter\.deduce_action_space\(self\.env\)\s*\n"
)

# Anchor B2 (in get_groot_observation): the terminating `return obs`
#   line — again, outside the if/elif/else chain that assigns the
#   language keys. We insert BEFORE it.
lang_obs_anchor = re.compile(
    r"(?P<indent>[ \t]+)return obs\s*\n"
)

def _lang_space_injection(indent: str) -> str:
    # Reassignment is idempotent — safe even if the else-branch already set it.
    return (
        f"{indent}# {LANG_MARKER} — expose task_description key for newer checkpoints\n"
        f"{indent}self.observation_space[\"annotation.human.action.task_description\"] = spaces.Text(\n"
        f"{indent}    max_length=256, charset=ALLOWED_LANGUAGE_CHARSET\n"
        f"{indent})\n"
    )

def _lang_obs_injection(indent: str) -> str:
    # raw_obs['language'] is the un-prefixed task description; the
    # "locked_waist: "/"unlocked_waist: " prefixes are added ONLY when
    # writing to the coarse_action key above. So just mirror it.
    # `setdefault` protects the pre-existing key when the else-branch
    # (non-GR1 robots) already set it.
    return (
        f"{indent}# {LANG_MARKER} — mirror raw language under task_description key\n"
        f"{indent}obs.setdefault(\"annotation.human.action.task_description\", raw_obs[\"language\"])\n"
    )

# ------------- APPLY -------------
inserts = []  # list of (start_offset, injection_text)

# Video alias — skip if already applied.
if VIDEO_MARKER in src:
    print(f"[patch]   video alias already present — skipping")
else:
    video_matches = list(video_anchor.finditer(src))
    if len(video_matches) < 2:
        sys.exit(
            f"PATCH FAILED (video): expected 2 anchor blocks, "
            f"found {len(video_matches)}. Edit by hand."
        )
    # A1 = first (in __init__), A2 = second (in get_groot_observation)
    inserts.append((video_matches[0].end(), _video_space_injection(video_matches[0].group("indent"))))
    inserts.append((video_matches[1].end(), _video_obs_injection(video_matches[1].group("indent"))))
    print(f"[patch]   queued video alias (2 sites)")

# Language alias — skip if already applied.
if LANG_MARKER in src:
    print(f"[patch]   language alias already present — skipping")
else:
    m_lspace = lang_space_anchor.search(src)
    m_lobs = lang_obs_anchor.search(src)
    if not m_lspace or not m_lobs:
        sys.exit(
            f"PATCH FAILED (language): could not find one or both anchors "
            f"(space={bool(m_lspace)}, obs={bool(m_lobs)}). Edit by hand."
        )
    # Insert BEFORE these anchors — we're placing sibling statements
    # outside the surrounding if/elif/else chain.
    inserts.append((m_lspace.start(), _lang_space_injection(m_lspace.group("indent"))))
    inserts.append((m_lobs.start(), _lang_obs_injection(m_lobs.group("indent"))))
    print(f"[patch]   queued language alias (2 sites)")

if not inserts:
    print(f"[patch] Nothing to do — file is already fully patched.")
    sys.exit(0)

# Apply in REVERSE offset order so earlier insertions don't shift later offsets.
for offset, text in sorted(inserts, key=lambda t: -t[0]):
    src = src[:offset] + text + src[offset:]

open(path, "w").write(src)
print(f"[patch] Patched {path}")
PY

echo ""
echo "[patch] verification (marker occurrences):"
grep -cE "PATCHED_VIDEO_EGO_VIEW_ALIAS|PATCHED_LANGUAGE_TASK_DESC_ALIAS" "$FILE" \
    | xargs -I{} echo "  markers present: {}"

echo ""
echo "[patch] Sanity import check:"
$PY - <<'PY' 2>/dev/null
import robocasa.utils.gym_utils.gymnasium_groot as g
src = open(g.__file__).read()
checks = [
    ('video obs alias',   'obs["video.ego_view"] = obs["video.ego_view_pad_res256_freq20"]'),
    ('video space alias', 'self.observation_space["video.ego_view"] = spaces.Box('),
    ('lang obs alias',    'obs.setdefault("annotation.human.action.task_description", raw_obs["language"])'),
    ('lang space alias',  'self.observation_space["annotation.human.action.task_description"] = spaces.Text('),
]
missing = [name for name, needle in checks if needle not in src]
if missing:
    raise SystemExit(f"  ✗ MISSING: {missing} — check {g.__file__} by hand.")
print(f"  ✓ all 4 aliases present in {g.__file__}")
PY

echo ""
echo "[patch] Done. Re-run the client — the language-key error should be gone."
echo ""
echo "To revert:  mv $FILE.bak $FILE"
