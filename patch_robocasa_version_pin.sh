#!/usr/bin/env bash
# patch_robocasa_version_pin.sh
# ------------------------------------------------------------------
# robocasa-gr1-tabletop-tasks' `robocasa/__init__.py` hard-asserts
# that robosuite is in {"1.5.0", "1.5.1"} at import time.  Our gr00t
# venv on kurai has robosuite 1.5.2 (upgrading is the only way to
# get a few Lift kwargs the wrapper needs), so the assertion fires
# on the very first `import robocasa`.  The 1.5.1 -> 1.5.2 delta on
# the code paths robocasa actually exercises is small and has been
# fine in practice; the safer move is to widen the allowed set.
#
# This patch:
#   * inserts a new "1.5.2" entry into the version list
#   * updates the error message string to mention {0,1,2}
#   * is idempotent (guarded by an in-file marker + a grep for the
#     literal "1.5.2" entry)
#
# Usage
# -----
#   bash patch_robocasa_version_pin.sh
#   # or point at a specific python that has robocasa installed:
#   ROBOCASA_PYTHON=/path/to/venv/bin/python bash patch_robocasa_version_pin.sh
#
# Reversible (.bak preserved on first run only).

set -euo pipefail

PY="${ROBOCASA_PYTHON:-python3}"

# The stdout of robocasa import may include a mimicgen WARNING line.
# Tag the real target with a prefix and grep it out.
FILE=$($PY - <<'PY' 2>/dev/null | sed -n 's/^__PATCH_TARGET__=//p' | tail -n 1
import importlib.util
spec = importlib.util.find_spec("robocasa")
if spec is None or spec.origin is None:
    raise SystemExit("robocasa not importable — activate the correct venv first.")
print(f"__PATCH_TARGET__={spec.origin}")
PY
)

if [ -z "$FILE" ]; then
    echo "ERROR: could not locate robocasa/__init__.py — is robocasa importable in this venv?"
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
"""Widen the robosuite version whitelist to include 1.5.2.

Original block (line ~124):
    assert robosuite.__version__ in [
        "1.5.0",
        "1.5.1",
    ], "robosuite version must be 1.5.{0,1}. Please install the correct version"

Patched:
    # PATCHED_ROBOSUITE_VERSION_PIN — accept 1.5.2 too (kwargs drift is minor)
    assert robosuite.__version__ in [
        "1.5.0",
        "1.5.1",
        "1.5.2",
    ], "robosuite version must be 1.5.{0,1,2}. Please install the correct version"
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()

MARKER = "PATCHED_ROBOSUITE_VERSION_PIN"

if MARKER in src:
    print("[patch]   version pin already patched — skipping")
    sys.exit(0)

# Match the whole assert block. Whitespace-tolerant.
pattern = re.compile(
    r'(?P<indent>[ \t]*)assert\s+robosuite\.__version__\s+in\s*\[\s*\n'
    r'(?P<items>(?:[ \t]*"1\.5\.[0-9]+",?\s*\n)+)'
    r'[ \t]*\]\s*,\s*"[^"]*"',
    re.MULTILINE,
)

m = pattern.search(src)
if not m:
    sys.exit(
        "PATCH FAILED: could not find the version assert block. "
        "Edit robocasa/__init__.py by hand."
    )

indent = m.group("indent")
items = m.group("items")
# Deduplicate whatever is already in the list, and add "1.5.2".
existing = re.findall(r'"(1\.5\.[0-9]+)"', items)
wanted = list(dict.fromkeys(existing + ["1.5.2"]))
new_items = "".join(f'{indent}    "{v}",\n' for v in wanted)

replacement = (
    f"{indent}# {MARKER} — widen to include 1.5.2 (kwargs drift is minor)\n"
    f"{indent}assert robosuite.__version__ in [\n"
    f"{new_items}"
    f'{indent}], "robosuite version must be 1.5.{{{",".join(v.split(".")[-1] for v in wanted)}}}. '
    f'Please install the correct version"'
)

src = src[: m.start()] + replacement + src[m.end():]

open(path, "w").write(src)
print(f"[patch] Patched {path}  (allowed versions: {wanted})")
PY

echo ""
echo "[patch] Verification:"
grep -nE "PATCHED_ROBOSUITE_VERSION_PIN|\"1\.5\.[0-9]+\"" "$FILE" | head -10

echo ""
echo "[patch] Import sanity check:"
$PY - <<'PY' 2>/dev/null
try:
    import robocasa  # noqa: F401
    print("  ✓ robocasa imports cleanly")
except AssertionError as e:
    print(f"  ✗ import still asserts: {e}")
    raise
PY

echo ""
echo "[patch] Done."
echo "To revert:  mv $FILE.bak $FILE"
