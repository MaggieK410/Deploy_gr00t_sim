#!/usr/bin/env bash
# patch_robocasa_kwargs.sh
# ------------------------------------------------------------------
# Fixes the version-drift TypeError that robocasa-gr1-tabletop-tasks
# raises when constructing any GR1 env:
#
#   TypeError: Lift.__init__() got an unexpected keyword argument 'seed'
#   TypeError: Lift.__init__() got an unexpected keyword argument 'translucent_robot'
#   (and possibly more, depending on version drift)
#
# Root cause: `robocasa/utils/gym_utils/gymnasium_basic.py` in the
# currently-pinned commit was written against a newer robosuite than
# the pinned `robosuite==1.5.1` accepts. Robocasa's `__init__.py` also
# hard-asserts robosuite in {1.5.0, 1.5.1}, so upgrading robosuite
# isn't an option either.
#
# Fix: introspect the target env class's `__init__` signature and drop
# any kwarg that isn't in it, right before `robosuite.make(**env_kwargs)`.
# One patch handles seed, translucent_robot, and any future kwarg the
# robocasa file adds ahead of robosuite.
#
# Usage
# -----
#   # Default path (matches kurai's install location):
#   bash patch_robocasa_kwargs.sh
#
#   # Or point at a custom robocasa install:
#   ROBOCASA_ROOT=/some/other/path bash patch_robocasa_kwargs.sh
#
# Idempotent: running it twice is safe (it detects the marker and no-ops).
# Reversible: leaves a `.bak` beside the patched file.

set -euo pipefail

DEFAULT_ROOT="/home/mkulcsar/robocasa-gr1-tabletop-tasks"
ROOT="${ROBOCASA_ROOT:-$DEFAULT_ROOT}"
FILE="$ROOT/robocasa/utils/gym_utils/gymnasium_basic.py"

if [ ! -f "$FILE" ]; then
    echo "ERROR: $FILE not found."
    echo "       Override with ROBOCASA_ROOT=/path/to/robocasa-gr1-tabletop-tasks"
    exit 1
fi

# Idempotency check — grep for the marker we insert.
if grep -q "PATCHED (version drift band-aid)" "$FILE"; then
    echo "[patch] $FILE already patched — no-op."
    grep -n "PATCHED" "$FILE" | head -3
    exit 0
fi

echo "[patch] Backing up $FILE -> $FILE.bak"
cp "$FILE" "$FILE.bak"

python3 - "$FILE" <<'PY'
"""Insert a signature-based kwargs filter right before
`env = robosuite.make(**env_kwargs)` in the given file."""
import re
import sys

path = sys.argv[1]
src = open(path).read()

# Patch body written WITHOUT any baseline indent — we apply the captured
# indent below. Two behaviors in one block:
#   (1) Filter env_kwargs to what the target env class accepts (band-aid
#       for robocasa passing newer robosuite kwargs like seed,
#       translucent_robot).
#   (2) Replace `env = robosuite.make(...)` with a try/retry loop that
#       catches `ValueError: No "camera" with name X exists`, substitutes
#       X with `robot0_robotview` (an available camera on the Lift scene),
#       and retries. Handles any future missing-camera name too.
FALLBACK_CAMERA = "robot0_robotview"

patch_lines = [
    "# ── PATCHED (version drift band-aid) ───────────────────────────────",
    "# (1) Filter env_kwargs to what the target env class accepts.",
    "#     robocasa was written against a newer robosuite that has extra",
    "#     kwargs (seed, translucent_robot, ...); robosuite@v1.5.1 doesn't.",
    "# (2) On missing-camera ValueError from robosuite.make, substitute the",
    f'#     missing camera name with "{FALLBACK_CAMERA}" and retry.',
    "import inspect as _insp",
    "import re as _re",
    "from robosuite.environments.base import REGISTERED_ENVS as _REG",
    "_en = env_kwargs.get(\"env_name\")",
    "if _en in _REG:",
    "    _cls = _REG[_en]",
    "    _accepted = set(_insp.signature(_cls.__init__).parameters.keys())",
    "    env_kwargs = {k: v for k, v in env_kwargs.items()",
    "                  if k in _accepted or k == \"env_name\"}",
    f'_FALLBACK_CAM = "{FALLBACK_CAMERA}"',
    "for _try in range(8):",
    "    try:",
    "        env = robosuite.make(**env_kwargs)",
    "        break",
    "    except ValueError as _e:",
    "        _emsg = str(_e)",
    "        # robosuite raises this in two forms depending on whether it's",
    "        # caught & wrapped by Observable._check_sensor_validity:",
    "        #   direct : 'No \"camera\" with name <name> exists'",
    "        #   wrapped: 'Current sensor for observable <name>_image is invalid.'",
    "        _mcam = (_re.search(r'No \"camera\" with name (\\S+) exists', _emsg)",
    "                 or _re.search(r'observable (\\S+?)_image is invalid', _emsg))",
    "        if _mcam and \"camera_names\" in env_kwargs:",
    "            _bad = _mcam.group(1)",
    "            _cams = list(env_kwargs[\"camera_names\"])",
    "            if _bad in _cams:",
    "                _idx = _cams.index(_bad)",
    "                if _FALLBACK_CAM in _cams:",
    "                    del _cams[_idx]",
    "                else:",
    "                    _cams[_idx] = _FALLBACK_CAM",
    "                print(f'[robocasa patch] camera \"{_bad}\" missing from '",
    "                      f'scene; substituting -> \"{_FALLBACK_CAM}\" '",
    "                      f'(result: {_cams})')",
    "                env_kwargs[\"camera_names\"] = _cams",
    "                continue",
    "        raise",
    "else:",
    "    raise RuntimeError('Failed after 8 camera-substitution retries')",
    "# ── END PATCH ──────────────────────────────────────────────────────",
]

# Find `env = robosuite.make(**env_kwargs)`, preserving its leading whitespace.
# We REPLACE that line entirely — our patch contains its own robosuite.make
# call inside the retry loop, so the original line must not run twice.
call_pat = re.compile(
    r"^(?P<indent>[ \t]*)env\s*=\s*robosuite\.make\(\*\*env_kwargs\)[ \t]*\n",
    re.MULTILINE,
)
m = call_pat.search(src)
if not m:
    sys.exit(
        "PATCH FAILED: could not find the line "
        "`env = robosuite.make(**env_kwargs)` in "
        f"{path}. Edit by hand."
    )

indent = m.group("indent")
indented_patch = "".join(indent + line + "\n" for line in patch_lines)

new_src = src[: m.start()] + indented_patch + src[m.end():]
open(path, "w").write(new_src)
print(f"[patch] Patched {path}")
PY

echo "[patch] Verification:"
grep -n "PATCHED\|robosuite.make(\*\*env_kwargs)" "$FILE" | head -6

echo ""
echo "[patch] Done. Try running the client again:"
echo "  python gr00t/eval/rollout_policy.py \\"
echo "      --n_episodes 1 \\"
echo "      --model_path \"\" \\"
echo "      --policy_client_host 127.0.0.1 --policy_client_port 5555 \\"
echo "      --max_episode_steps 720 \\"
echo "      --env_name gr1_unified/Lift_GR1ArmsAndWaistFourierHands_Env \\"
echo "      --n_action_steps 16 --n_envs 1"
echo ""
echo "To revert:  mv $FILE.bak $FILE"
