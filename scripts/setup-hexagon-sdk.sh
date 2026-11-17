#!/bin/bash -e
#
# Downloads the Hexagon SDK into $HEXAGON_INSTALL_DIR (default ~/.hexagon-sdk),
# where android/build.gradle and scripts/build-android.sh look for it. Only
# needed to build the rnwhisper_*_hexagon (Qualcomm NPU) variant from source;
# the prebuilt libraries already include it. The DSP-side libraries are a
# separate step (scripts/build-hexagon-htp.sh, Docker).
#
# On macOS the SDK tools do not run, but the headers are all the host-side
# build needs.
#
# --headers-only keeps just those headers (the SDK directories that
# android/src/main/rnwhisper/CMakeLists.txt includes): under 1 MB instead of
# 3 GB, which CI caches. It is not enough for scripts/build-hexagon-htp.sh
# without Docker, which runs the SDK's tools.
HEADER_DIRS=(incs ipc/fastrpc/rpcmem/inc)

HEADERS_ONLY=0
if [ "${1:-}" = "--headers-only" ]; then
  HEADERS_ONLY=1
elif [ $# -gt 0 ]; then
  echo "Usage: $0 [--headers-only]" >&2
  exit 1
fi

HEXAGON_SDK_VERSION="${HEXAGON_SDK_VERSION:-6.4.0.2}"
HEXAGON_INSTALL_DIR="${HEXAGON_INSTALL_DIR:-$HOME/.hexagon-sdk}"
SDK_DIR="$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"

if [ -d "$SDK_DIR" ]; then
  if [ "$HEADERS_ONLY" = "1" ] || [ -d "$SDK_DIR/tools" ]; then
    echo "Hexagon SDK already installed: $SDK_DIR"
    exit 0
  fi
  echo "Completing the headers-only install at $SDK_DIR with the full SDK"
fi

URL="https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v${HEXAGON_SDK_VERSION}/hexagon-sdk-v${HEXAGON_SDK_VERSION}-amd64-lnx.tar.xz"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

echo "Downloading Hexagon SDK v${HEXAGON_SDK_VERSION}..."
curl -L --fail -o "$TEMP_DIR/hexagon-sdk.tar.xz" "$URL"

echo "Extracting to $HEXAGON_INSTALL_DIR..."
mkdir -p "$HEXAGON_INSTALL_DIR"
tar -xaf "$TEMP_DIR/hexagon-sdk.tar.xz" -C "$HEXAGON_INSTALL_DIR"

if [ ! -d "$SDK_DIR" ]; then
  echo "Error: expected $SDK_DIR after extraction; contents of $HEXAGON_INSTALL_DIR:"
  ls "$HEXAGON_INSTALL_DIR"
  exit 1
fi

if [ "$HEADERS_ONLY" = "1" ]; then
  echo "Keeping only the headers: ${HEADER_DIRS[*]}"
  HEADERS_DIR="$HEXAGON_INSTALL_DIR/.headers-$HEXAGON_SDK_VERSION"
  rm -rf "$HEADERS_DIR"
  for dir in "${HEADER_DIRS[@]}"; do
    mkdir -p "$(dirname "$HEADERS_DIR/$dir")"
    mv "$SDK_DIR/$dir" "$HEADERS_DIR/$dir"
  done
  rm -rf "$SDK_DIR"
  mv "$HEADERS_DIR" "$SDK_DIR"
fi

echo "Hexagon SDK installed to $SDK_DIR"
