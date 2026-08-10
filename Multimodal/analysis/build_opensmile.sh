#!/data/data/com.termux/files/usr/bin/bash
# build_opensmile.sh — Build openSMILE 3.x from source on Termux (S24 Ultra)
#
# Outputs:
#   ~/.local/bin/SMILExtract       # the standalone CLI binary
#   ~/opensmile-src/config/...     # full config tree (eGeMAPSv02.conf, etc.)
#
# Wall-clock: ~20-40 min on Snapdragon 8 Gen 3 with `make -j$(nproc)`.
# Re-runnable: skips clone if ~/opensmile-src/ already exists.
#
# Requires (install first):
#   pkg install -y cmake make clang git
set -euo pipefail

PREFIX="$HOME/.local"
SRC="$HOME/opensmile-src"
REPO_URL="https://github.com/audeering/opensmile.git"

mkdir -p "$PREFIX/bin"

echo "============================================================="
echo " build_opensmile.sh — openSMILE 3.x on Termux"
echo "============================================================="
echo "PREFIX = $PREFIX"
echo "SRC    = $SRC"
echo ""

# Dep check
for cmd in cmake make clang git; do
    if ! command -v "$cmd" >/dev/null; then
        echo "❌ Missing required tool: $cmd"
        echo "   Run: pkg install -y cmake make clang git"
        exit 1
    fi
done
echo "✓ cmake $(cmake --version | head -1)"
echo "✓ clang $(clang --version | head -1)"
echo "✓ make  $(make --version | head -1)"

# Termux + CMake 4.x quirk: CMake's Android system detection reads
# ${CMAKE_INSTALL_PREFIX}/include/android/api-level.h. With our PREFIX
# pointing at ~/.local, the header (which lives in $PREFIX/include/android/)
# is not visible. Pre-create the symlink so the configure step succeeds.
# See: https://github.com/termux/termux-packages/issues/17104
TERMUX_PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
if [[ -f "$TERMUX_PREFIX/include/android/api-level.h" \
      && ! -e "$HOME/.local/include/android/api-level.h" ]]; then
    mkdir -p "$HOME/.local/include/android"
    ln -sf "$TERMUX_PREFIX/include/android/api-level.h" \
           "$HOME/.local/include/android/api-level.h"
    echo "✓ symlinked android/api-level.h into ~/.local/include/android/ (CMake-4 Termux quirk)"
fi

# Clone (idempotent)
if [[ ! -d "$SRC" ]]; then
    echo ""
    echo "[1/3] Cloning openSMILE into $SRC ..."
    git clone --depth=1 --recursive "$REPO_URL" "$SRC"
else
    echo ""
    echo "[1/3] $SRC already exists — skipping clone"
fi

# Configure
echo ""
echo "[2/3] cmake ..."
cd "$SRC"
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_PYTHON=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DSTATIC_LINK=OFF \
    -DCMAKE_SHARED_LINKER_FLAGS="-llog" \
    -DCMAKE_EXE_LINKER_FLAGS="-llog"
# NOTE: -llog links against Termux/Android's liblog.so which provides
# __android_log_print. openSMILE's src/android/openslesSource.cpp uses
# it; without this flag, the final SMILExtract link fails with
#   ld.lld: error: undefined reference: __android_log_print
# (disallowed by --no-allow-shlib-undefined).

# Build
NPROC="$(nproc 2>/dev/null || echo 2)"
echo ""
echo "[3/3] make -j${NPROC} SMILExtract ..."
echo "      This is the slow step (~20-40 min on S24 Ultra). Patience."
make -j"${NPROC}" SMILExtract

# Locate produced binary (path differs across openSMILE versions)
BIN=""
for cand in \
    "$SRC/build/progsrc/smilextract/SMILExtract" \
    "$SRC/build/SMILExtract" \
    "$SRC/build/bin/SMILExtract"; do
    if [[ -x "$cand" ]]; then BIN="$cand"; break; fi
done
if [[ -z "$BIN" ]]; then
    echo "❌ Build finished but SMILExtract binary not found in expected locations."
    echo "   Search: find $SRC/build -name SMILExtract -type f"
    exit 1
fi

cp "$BIN" "$PREFIX/bin/SMILExtract"
chmod +x "$PREFIX/bin/SMILExtract"

# Install libopensmile.so next to the binary so the dynamic linker
# finds it (SMILExtract depends on this .so at runtime).
SO_SRC=""
for cand in \
    "$SRC/build/libopensmile.so" \
    "$SRC/build/libopensmile.so.0" \
    "$SRC/build/src/libopensmile.so"; do
    if [[ -e "$cand" ]]; then SO_SRC="$cand"; break; fi
done
if [[ -n "$SO_SRC" ]]; then
    mkdir -p "$PREFIX/lib"
    cp "$SO_SRC" "$PREFIX/lib/"
    echo "✓ copied $(basename "$SO_SRC") → $PREFIX/lib/"
fi

# Ensure ~/.local/lib is on LD_LIBRARY_PATH in future shells
if ! grep -q 'HOME/.local/lib' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"' >> "$HOME/.bashrc"
    echo "✓ added LD_LIBRARY_PATH=~/.local/lib to ~/.bashrc"
fi

# Optionally also install the standalone eGeMAPSv02 config to a stable path
CONF_SRC="$SRC/config/egemaps/v02/eGeMAPSv02.conf"
if [[ -f "$CONF_SRC" ]]; then
    mkdir -p "$PREFIX/share/opensmile"
    cp -r "$SRC/config" "$PREFIX/share/opensmile/"
fi

# Verify
echo ""
echo "============================================================="
echo " Done."
echo "============================================================="
echo "SMILExtract binary: $PREFIX/bin/SMILExtract"
echo "eGeMAPSv02 config : $SRC/config/egemaps/v02/eGeMAPSv02.conf"
"$PREFIX/bin/SMILExtract" -h 2>&1 | head -3 || true
echo ""
echo "Ensure ~/.local/bin is on PATH:"
echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
echo "  source ~/.bashrc"
