#!/usr/bin/env bash
# patch_robosuite_lift_kwargs.sh
# ------------------------------------------------------------------
# Adds **_extra_kwargs to `Lift.__init__` in robosuite so it silently
# accepts (and discards) kwargs that robocasa-gr1-tabletop-tasks passes
# but the pinned robosuite@v1.5.1 doesn't natively support:
#
#   TypeError: Lift.__init__() got an unexpected keyword argument 'seed'
#   TypeError: Lift.__init__() got an unexpected keyword argument 'translucent_robot'
#
# The PnP env classes in this robosuite already accept **kwargs (which
# is why PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env works).
# Lift is the outlier. This patch aligns Lift with the PnP convention.
#
# Rationale over patching robocasa: the robocasa-side kwargs filter in
# `patch_robocasa_kwargs.sh` is a fine fix too, but this one is more
# targeted — it changes ONE line in Lift.__init__ instead of injecting
# a signature-filter block, and it only affects the Lift env class.
#
# Usage
# -----
#   # Default: auto-detect robosuite's install path via importable module
#   bash patch_robosuite_lift_kwargs.sh
#
#   # Or point at a specific python that has robosuite installed
#   ROBOSUITE_PYTHON=/path/to/venv/bin/python bash patch_robosuite_lift_kwargs.sh
#
# Idempotent (marker check). Reversible (leaves a .bak).

set -euo pipefail

PY="${ROBOSUITE_PYTHON:-python3}"

FILE=$($PY - <<'PY'
import importlib.util
spec = importlib.util.find_spec("robosuite.environments.manipulation.lift")
if spec is None or spec.origin is None:
    raise SystemExit("robosuite.environments.manipulation.lift not importable — "
                     "activate the correct venv first.")
print(spec.origin)
PY
)

echo "[patch] target: $FILE"

if grep -q "PATCHED_LIFT_KWARGS_SINK" "$FILE"; then
    echo "[patch] $FILE already patched — no-op."
    grep -n "PATCHED_LIFT_KWARGS_SINK" "$FILE" | head -3
    exit 0
fi

echo "[patch] backing up $FILE -> $FILE.bak"
cp "$FILE" "$FILE.bak"

$PY - "$FILE" <<'PY'
"""Insert `**_extra_kwargs` at the end of the Lift class's `__init__`
parameter list, so it silently accepts any unknown kwargs. Marker
comment appended so we can detect re-runs.
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()

# Robosuite Lift class __init__ signature spans multiple lines. Match:
#     def __init__(
#         self,
#         robots,
#         env_configuration=...,
#         ...  (many lines) ...
#     ):
# We inject `**_extra_kwargs,` as the LAST param, right before the closing `):`.
sig_pat = re.compile(
    r"(class\s+Lift\b[^\n]*:.*?)"                     # class header + body up to __init__
    r"(def\s+__init__\s*\(\s*self,)"                  # start of __init__
    r"(.*?)"                                          # rest of params
    r"(\n\s*)(\))"                                    # closing paren
    r"(:\s*)",                                        # colon
    flags=re.DOTALL,
)
m = sig_pat.search(src)
if not m:
    sys.exit(
        "PATCH FAILED: could not locate Lift.__init__ signature in "
        f"{path}. Edit by hand."
    )

# Params section (m.group(3)) ends with a trailing comma or not.
params = m.group(3)
if not params.rstrip().endswith(","):
    params = params.rstrip() + ","

# Insert **_extra_kwargs on its own line at the same indent as `self,`
new_params = params + "\n        **_extra_kwargs,  # PATCHED_LIFT_KWARGS_SINK"

new = (
    src[: m.start()]
    + m.group(1)
    + m.group(2)
    + new_params
    + m.group(4)
    + m.group(5)
    + m.group(6)
    + src[m.end():]
)
open(path, "w").write(new)
print(f"[patch] Patched Lift.__init__ in {path}")
PY

echo "[patch] verification:"
grep -n "PATCHED_LIFT_KWARGS_SINK\|def __init__" "$FILE" | head -10

echo ""
echo "[patch] Sanity import check:"
$PY - <<'PY'
import robosuite
from robosuite.environments.manipulation.lift import Lift
import inspect
sig = inspect.signature(Lift.__init__)
params = list(sig.parameters.keys())
print("  Lift.__init__ params (last 5):", params[-5:])
assert "_extra_kwargs" in params, "PATCH DID NOT LAND — check the file by hand."
print("  ✓ **_extra_kwargs is present")
PY

echo ""
echo "[patch] Done. Try running the client:"
echo "  python gr00t/eval/rollout_policy.py \\"
echo "      --n_episodes 1 --model_path \"\" \\"
echo "      --policy_client_host 127.0.0.1 --policy_client_port 5555 \\"
echo "      --max_episode_steps 720 \\"
echo "      --env_name gr1_unified/Lift_GR1ArmsAndWaistFourierHands_Env \\"
echo "      --n_action_steps 16 --n_envs 1"
echo ""
echo "To revert:  mv $FILE.bak $FILE"
