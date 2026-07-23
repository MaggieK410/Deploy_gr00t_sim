#!/usr/bin/env bash
# patch_rollout_policy_episode_ids.sh
# ------------------------------------------------------------------
# Injects a per-slot episode counter into the observation dict on the
# client side of `gr00t/eval/rollout_policy.py`, so `custom_sim_server.py`
# can group its attention captures by episode without any state-based
# reset detection.
#
# The rollout loop knows episode boundaries precisely — it sees the
# `terminations[env_idx]` / `truncations[env_idx]` flags every step.
# This patch:
#
#   (a) Right after `observations, _ = env.reset()`, initialize
#       `_slot_ep_ids = np.arange(n_envs, dtype=np.int32)` and
#       `_next_ep_id = n_envs`.
#
#   (b) Right before `actions, _ = policy.get_action(observations)`,
#       stuff `observations["slot_ep_ids"] = _slot_ep_ids.copy()` so
#       the ZMQ payload carries the current per-slot ids.
#
#   (c) In the termination/truncation branch of the per-env-idx loop,
#       when an episode ends on slot `env_idx`, do:
#           _slot_ep_ids[env_idx] = _next_ep_id
#           _next_ep_id += 1
#       so the NEXT observation the policy sees carries a fresh id
#       for that slot (matching the newly auto-reset env).
#
# All three injections are guarded by an in-body marker so the script
# is idempotent — re-running it does nothing if the patch is already
# present.
#
# Usage
# -----
#   ROLLOUT_PATH=/path/to/rollout_policy.py bash patch_rollout_policy_episode_ids.sh
#   # or, if it's importable:
#   ROLLOUT_PYTHON=/path/to/venv/bin/python bash patch_rollout_policy_episode_ids.sh
#
# Reversible (.bak preserved on first run only).

set -euo pipefail

PY="${ROLLOUT_PYTHON:-python3}"

if [ -n "${ROLLOUT_PATH:-}" ]; then
    FILE="$ROLLOUT_PATH"
else
    FILE=$($PY - <<'PY' 2>/dev/null | sed -n 's/^__PATCH_TARGET__=//p' | tail -n 1
import importlib.util
spec = importlib.util.find_spec("gr00t.eval.rollout_policy")
if spec is None or spec.origin is None:
    raise SystemExit("gr00t.eval.rollout_policy not importable — activate the gr00t venv first.")
print(f"__PATCH_TARGET__={spec.origin}")
PY
)
fi

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    echo "ERROR: could not locate rollout_policy.py. Pass ROLLOUT_PATH=<path> or activate the gr00t venv."
    exit 1
fi

echo "[patch] target: $FILE"

if [ ! -f "$FILE.bak" ]; then
    echo "[patch] backing up $FILE -> $FILE.bak"
    cp "$FILE" "$FILE.bak"
else
    echo "[patch] $FILE.bak already exists — leaving it alone (represents pristine file)"
fi

$PY - "$FILE" <<'PY'
"""Three injections in rollout_policy.py, guarded by one marker."""
import re
import sys

path = sys.argv[1]
src = open(path).read()

MARKER = "PATCHED_ROLLOUT_SLOT_EP_IDS"

if MARKER in src:
    print("[patch]   slot_ep_ids injection already present — skipping")
    sys.exit(0)

# ---- Injection (a): after `observations, _ = env.reset()` ----
init_anchor = re.compile(
    r"(?P<indent>[ \t]+)observations, _ = env\.reset\(\)\s*\n"
)
m_init = init_anchor.search(src)
if not m_init:
    sys.exit("PATCH FAILED: could not find `observations, _ = env.reset()`")

init_indent = m_init.group("indent")
init_injection = (
    f"{init_indent}# {MARKER} — per-slot episode counter for custom_sim_server.py\n"
    f"{init_indent}_slot_ep_ids = np.arange(n_envs, dtype=np.int32)\n"
    f"{init_indent}_next_ep_id = int(n_envs)\n"
)

# ---- Injection (b): before `actions, _ = policy.get_action(observations)` ----
getaction_anchor = re.compile(
    r"(?P<indent>[ \t]+)actions, _ = policy\.get_action\(observations\)\s*\n"
)
m_get = getaction_anchor.search(src)
if not m_get:
    sys.exit("PATCH FAILED: could not find `actions, _ = policy.get_action(observations)`")

get_indent = m_get.group("indent")
inject_before_get = (
    f"{get_indent}# {MARKER} — stamp current per-slot episode ids into obs\n"
    f'{get_indent}observations["slot_ep_ids"] = _slot_ep_ids.copy()\n'
)

# ---- Injection (c): inside the termination/truncation branch ----
# We look for the exact line where the episode counter is incremented
# in the else-branch (`completed_episodes += 1`) — that's the sibling
# spot where we should bump `_slot_ep_ids[env_idx]` too.  We insert
# just above the `if terminations[env_idx] or truncations[env_idx]:`
# so the fresh id is assigned before the loop touches next_obs.
term_anchor = re.compile(
    r"(?P<indent>[ \t]+)if terminations\[env_idx\] or truncations\[env_idx\]:\s*\n"
)
# We want to insert an id-bump inside this if-block.  Look for the
# closing `current_lengths[env_idx] = 0` line, which is at the block's
# tail, and insert right after it (still inside the if).
tail_anchor = re.compile(
    r"(?P<indent>[ \t]+)current_lengths\[env_idx\] = 0\s*\n"
)
m_tail = tail_anchor.search(src)
if not m_tail:
    sys.exit("PATCH FAILED: could not find `current_lengths[env_idx] = 0`")

tail_indent = m_tail.group("indent")
# Cap the counter at n_episodes.  Any extra terminations after we've
# already handed out n_episodes ep_ids get a sentinel (-1) which the
# server drops.  This means we never save more than n_episodes files
# even though the async vec envs may have started more physical
# episodes in parallel.
inject_after_tail = (
    f"{tail_indent}# {MARKER} — assign a fresh ep_id if we still need more\n"
    f"{tail_indent}#            episodes; otherwise mark the slot as excess.\n"
    f"{tail_indent}if _next_ep_id < n_episodes:\n"
    f"{tail_indent}    _slot_ep_ids[env_idx] = _next_ep_id\n"
    f"{tail_indent}    _next_ep_id += 1\n"
    f"{tail_indent}else:\n"
    f"{tail_indent}    _slot_ep_ids[env_idx] = -1  # sentinel: server ignores\n"
)

# Apply in reverse offset order so earlier insertions don't shift later ones.
inserts = [
    (m_init.end(), init_injection),
    (m_get.start(), inject_before_get),
    (m_tail.end(), inject_after_tail),
]
for offset, text in sorted(inserts, key=lambda t: -t[0]):
    src = src[:offset] + text + src[offset:]

open(path, "w").write(src)
print(f"[patch] Patched {path} (3 sites)")
PY

echo ""
echo "[patch] Verification:"
grep -nE "PATCHED_ROLLOUT_SLOT_EP_IDS" "$FILE" | head -6

echo ""
echo "[patch] Import sanity check:"
$PY - <<'PY' 2>/dev/null
import gr00t.eval.rollout_policy  # noqa: F401
print("  ✓ gr00t.eval.rollout_policy imports cleanly")
PY

echo ""
echo "[patch] Done."
echo "To revert:  mv $FILE.bak $FILE"
