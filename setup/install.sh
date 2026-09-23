#!/usr/bin/env bash
# ===============================================================
#  Quant Research Kit - one-click environment setup (Linux / macOS)
#
#  Creates a .venv inside the project and installs the full
#  VeighNa + Qlib + akshare stack.
# ===============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
VENV="$PROJ/.venv"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo
echo "================================================================"
echo "  Quant Research Kit - environment setup"
echo "================================================================"
echo "  project : $PROJ"
echo "  venv    : $VENV"
echo "  mirror  : $MIRROR"
echo

# ---- 1. locate a suitable interpreter (3.10 - 3.12) ----
SYSPY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys; sys.exit(0 if (3,10)<=sys.version_info<(3,13) else 1)' 2>/dev/null; then
            SYSPY="$cand"
            break
        fi
    fi
done
if [ -z "$SYSPY" ]; then
    echo "[ERROR] No suitable Python 3.10-3.12 found."
    echo "        Python 3.13+ is not supported: vnpy_riskmanager has no matching wheel."
    exit 1
fi
echo "[1/4] interpreter: $SYSPY ($($SYSPY -c 'import sys;print(sys.version.split()[0])'))"

# ---- 2. create venv ----
if [ -x "$VENV/bin/python" ]; then
    echo "[2/4] venv already exists, reusing it"
else
    echo "[2/4] creating venv ..."
    "$SYSPY" -m venv "$VENV"
fi
PY="$VENV/bin/python"

echo "[3/4] upgrading pip ..."
"$PY" -m pip install -q -U pip -i "$MIRROR"

echo "[4/4] installing packages (this downloads ~400MB, be patient) ..."
"$PY" -m pip install -i "$MIRROR" -r "$PROJ/setup/requirements.txt"

echo
echo "================================================================"
echo "  Done.  Next step:"
echo
echo "    $PY scripts/00_check_env.py"
echo
echo "  Expect all 15 checks to pass. If something is missing, see the"
echo "  quant-env-inventory skill - the package may live in another"
echo "  interpreter on this machine."
echo "================================================================"
