#!/usr/bin/env bash
# Bootstrap deps on a fresh Jetson Nano clone (Python 3.6.9, JetPack 4.6.x).
#
# Why a script and not `pip install -r requirements.txt`:
#   pycuda's sdist declares `setup_requires=['numpy']`. pip runs `setup.py
#   egg_info` on every requirement *before* installing any of them, so numpy
#   is not yet in the env when pycuda's egg_info fires. setuptools then
#   falls back to easy_install, fetches the latest numpy sdist (PEP 517,
#   no setup.py), and dies. The fix is to install numpy *first*, then the
#   rest, then pycuda. Doing this in a script also lets us set CUDA env
#   vars and pin build tools in one place.
#
# Usage (from repo root, inside a venv created with --system-site-packages
# so cv2 + tensorrt from JetPack are visible):
#   bash tools/install_jetson.sh

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# --- CUDA on PATH ----------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    export CUDA_HOME="/usr/local/cuda"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Expected at /usr/local/cuda/bin/nvcc." >&2
    echo "       Check 'ls /usr/local/' — JetPack may have installed CUDA" >&2
    echo "       under /usr/local/cuda-10.2/ without a /usr/local/cuda symlink." >&2
    exit 1
fi
echo "[install] using $(nvcc --version | tail -n1)"

# --- apt: protoc for onnx's C++ build (onnx has no aarch64/Py3.6 wheel) ----
# onnx in requirements.txt builds from sdist on Jetson; its CMake config
# requires `protoc` from system protobuf. If protoc is already installed
# (or onnx was dropped from requirements.txt), this is a no-op.
if ! command -v protoc >/dev/null 2>&1; then
    if [ "$(id -u)" = "0" ]; then
        sudo_cmd=""
    elif command -v sudo >/dev/null 2>&1; then
        sudo_cmd="sudo"
    else
        echo "WARN: protoc missing and no sudo available — onnx build will fail." >&2
        echo "      Install manually: apt-get install -y protobuf-compiler libprotoc-dev libprotobuf-dev" >&2
        sudo_cmd=""
    fi
    echo "[install] installing protobuf-compiler via apt (needed by onnx sdist build)"
    DEBIAN_FRONTEND=noninteractive ${sudo_cmd} apt-get install -y \
        protobuf-compiler libprotoc-dev libprotobuf-dev
fi

# --- pip flag: --user outside venv, plain inside ---------------------------
pip_flags=()
if [ -z "${VIRTUAL_ENV:-}" ]; then
    pip_flags=(--user)
    echo "[install] no venv detected — installing with --user"
else
    echo "[install] venv: $VIRTUAL_ENV"
fi

pip() { python -m pip "$@"; }

# --- 1. build tools that still work on Py3.6 -------------------------------
# pip >= 22 / setuptools >= 60 drop Py3.6 or break pycuda sdist.
# Cython is needed because numpy 1.19.5 has no aarch64/Py3.6 wheel on PyPI,
# so pip falls back to building from sdist which cythonizes during install.
echo "[install] upgrading pip / setuptools / wheel / Cython (Py3.6-compatible pins)"
pip install --no-cache-dir "${pip_flags[@]}" \
    'pip<21' 'setuptools<60' 'wheel' 'Cython<3'

# --- 2. numpy first, so pycuda's setup_requires is satisfied ---------------
echo "[install] installing numpy (must precede pycuda)"
pip install --no-cache-dir "${pip_flags[@]}" 'numpy>=1.19,<1.20'

# --- 2b. MarkupSafe pre-install (Py3.6 aarch64 only) ----------------------
# MarkupSafe 2.x has no aarch64 wheel for Py3.6, so pip falls back to its
# sdist. The sdist's setup.cfg uses `version = attr: markupsafe.__version__`,
# which makes setuptools try to `import markupsafe` during egg_info — before
# the package is installed — and the install fails. MarkupSafe 1.1.1 has an
# aarch64 cp36 wheel on PyPI, so pinning <2 makes pip pick the wheel and
# skip the sdist build entirely. MarkupSafe is pulled transitively (likely
# via Jinja2 in one of the deps), so we install it ahead of requirements.txt
# the same way numpy goes ahead of pycuda above.
echo "[install] installing MarkupSafe<2 (Py3.6 aarch64 has no MarkupSafe 2.x wheel)"
pip install --no-cache-dir "${pip_flags[@]}" 'MarkupSafe<2'

# --- 3. rest of project deps ----------------------------------------------
echo "[install] installing -r requirements.txt"
pip install --no-cache-dir "${pip_flags[@]}" -r requirements.txt

# --- 4. pycuda last -------------------------------------------------------
echo "[install] installing pycuda"
pip install --no-cache-dir "${pip_flags[@]}" 'pycuda<2022.0'

# --- 5. sanity check ------------------------------------------------------
echo "[install] verifying imports"
python - <<'PY'
import sys
required = ["numpy", "yaml", "psutil", "rich", "dataclasses"]
jetson_only = ["tensorrt", "cv2", "pycuda.driver"]
fail = []
for mod in required + jetson_only:
    try:
        __import__(mod)
        print("  ok    %s" % mod)
    except Exception as e:
        print("  FAIL  %s: %s" % (mod, e))
        fail.append(mod)
if fail:
    print()
    print("Some imports failed. If tensorrt/cv2 are missing, your venv was")
    print("created without --system-site-packages — recreate it with:")
    print("    python3 -m venv --system-site-packages venv")
    sys.exit(1)
PY

echo "[install] done. next: python tools/health_check.py --config config.yaml"
