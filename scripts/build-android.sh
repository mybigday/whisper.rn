#!/bin/bash -e
#
# Builds the prebuilt rnwhisper core libraries (librnwhisper*.so, one per
# CPU-feature variant) for every Android ABI into android/src/main/jniLibs/.
# These have no React Native dependency; the JNI/JSI wrapper is compiled by the
# consuming app's Gradle build against them (android/src/main/CMakeLists.txt).
#
# The release workflow packages the output as whisper-rn-android-jni-libs.tar.gz
# and the package's postinstall downloads it (install/download-native-artifacts.js).
#
# Hexagon: the rnwhisper_*_hexagon variant is built when the Hexagon SDK is
# found (scripts/setup-hexagon-sdk.sh) and scripts/build-hexagon-htp.sh has
# generated the FastRPC stub. Set RNWHISPER_REQUIRE_HEXAGON=1 to fail instead
# of silently building CPU-only (the release build does).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NDK_VERSION=27.3.13750724
ANDROID_PLATFORM=android-23
CMAKE_BUILD_TYPE=Release
ABIS=(${RNWHISPER_ANDROID_ABIS:-arm64-v8a armeabi-v7a x86_64 x86})

HEXAGON_SDK_VERSION="6.4.0.2"
HEXAGON_INSTALL_DIR="${HEXAGON_INSTALL_DIR:-$HOME/.hexagon-sdk}"

# Auto-detect the Android SDK/NDK location if not set
if [ -z "$ANDROID_HOME" ]; then
  for location in "/opt/android-sdk" "/opt/android" "/android-sdk" "$HOME/Android/Sdk" "$HOME/Library/Android/sdk"; do
    if [ -d "$location" ]; then
      export ANDROID_HOME="$location"
      echo "Auto-detected ANDROID_HOME: $ANDROID_HOME"
      break
    fi
  done

  if [ -z "$ANDROID_HOME" ]; then
    echo "Error: ANDROID_HOME not set and could not auto-detect the Android SDK"
    exit 1
  fi
fi

# ANDROID_HOME may point at a standalone NDK or at an SDK with ndk/<version>
if [ -f "$ANDROID_HOME/build/cmake/android.toolchain.cmake" ]; then
  NDK_DIR="$ANDROID_HOME"
  echo "Using standalone Android NDK: $NDK_DIR"
elif [ -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
  NDK_DIR="$ANDROID_HOME/ndk/$NDK_VERSION"
  echo "Using Android NDK: $NDK_DIR"
elif [ -d "$ANDROID_HOME/ndk" ] && [ -n "$(ls "$ANDROID_HOME/ndk" 2>/dev/null)" ]; then
  NDK_VERSION=$(ls "$ANDROID_HOME/ndk" | sort -V | tail -n 1)
  NDK_DIR="$ANDROID_HOME/ndk/$NDK_VERSION"
  echo "NDK $NDK_VERSION not found; using latest available: $NDK_DIR"
else
  echo "Error: NDK not found. Expected either:"
  echo "  - Standalone NDK: $ANDROID_HOME/build/cmake/android.toolchain.cmake"
  echo "  - SDK with NDK:   $ANDROID_HOME/ndk/$NDK_VERSION/"
  exit 1
fi

CMAKE_TOOLCHAIN_FILE="$NDK_DIR/build/cmake/android.toolchain.cmake"
STRIP=$(ls "$NDK_DIR"/toolchains/llvm/prebuilt/*/bin/llvm-strip | head -n 1)

CMAKE_PATH=$(command -v cmake || true)
if [ -z "$CMAKE_PATH" ] && [ -d "$ANDROID_HOME/cmake" ]; then
  VERSION=$(ls "$ANDROID_HOME/cmake" | grep -E "3\.[0-9]+\.[0-9]+" | sort -V | tail -n 1)
  if [ -n "$VERSION" ]; then
    CMAKE_PATH="$ANDROID_HOME/cmake/$VERSION/bin/cmake"
  fi
fi
if [ -z "$CMAKE_PATH" ]; then
  echo "Error: cmake could not be found, please install it"
  exit 1
fi

n_cpu=1
if uname -a | grep -q "Darwin"; then
  n_cpu=$(sysctl -n hw.logicalcpu)
elif uname -a | grep -q "Linux"; then
  n_cpu=$(nproc)
fi

# Hexagon SDK: the rnwhisper_*_hexagon variant only compiles the host side
# against the SDK headers, so the SDK is enough (no Hexagon tools needed here).
if [ -z "$HEXAGON_SDK_ROOT" ] && [ -d "$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION" ]; then
  export HEXAGON_SDK_ROOT="$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
fi

HEXAGON_STUB_DIR="$ROOT_DIR/vendor/whisper.cpp/ggml/src/ggml-hexagon/htp/v73"
HEXAGON_ENABLED=0
if [ -n "$HEXAGON_SDK_ROOT" ] && [ -d "$HEXAGON_SDK_ROOT/incs/stddef" ]; then
  if [ -f "$HEXAGON_STUB_DIR/htp_iface_stub.c" ] && [ -f "$HEXAGON_STUB_DIR/htp_iface.h" ]; then
    HEXAGON_ENABLED=1
    echo "Hexagon SDK found at $HEXAGON_SDK_ROOT - building the Hexagon variant"
  else
    echo "Hexagon SDK found but the FastRPC stub is missing; run scripts/build-hexagon-htp.sh first"
  fi
else
  echo "Hexagon SDK not found (scripts/setup-hexagon-sdk.sh) - building CPU-only"
fi

if [ "$HEXAGON_ENABLED" != "1" ] && [ "${RNWHISPER_REQUIRE_HEXAGON:-0}" = "1" ]; then
  echo "Error: RNWHISPER_REQUIRE_HEXAGON=1 but the Hexagon variant cannot be built"
  exit 1
fi

t0=$(date +%s)

cd "$ROOT_DIR/android/src/main/rnwhisper"
JNI_LIBS_DIR="$ROOT_DIR/android/src/main/jniLibs"

for ABI in "${ABIS[@]}"; do
  echo ""
  echo "Building $ABI prebuilt shared libraries..."
  BUILD_DIR="build-$ABI"
  rm -rf "$BUILD_DIR"

  EXTRA_ARGS=()
  if [ "$ABI" = "arm64-v8a" ] && [ "$HEXAGON_ENABLED" = "1" ]; then
    EXTRA_ARGS+=("-DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT")
  fi

  "$CMAKE_PATH" \
    -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DANDROID_STL=c++_shared \
    -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    "${EXTRA_ARGS[@]}" \
    -B "$BUILD_DIR"

  "$CMAKE_PATH" --build "$BUILD_DIR" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"

  mkdir -p "$JNI_LIBS_DIR/$ABI"
  rm -f "$JNI_LIBS_DIR/$ABI"/librnwhisper*.so
  for lib in "$BUILD_DIR"/librnwhisper*.so; do
    echo "Stripping $(basename "$lib")..."
    "$STRIP" "$lib"
    cp "$lib" "$JNI_LIBS_DIR/$ABI/"
  done

  rm -rf "$BUILD_DIR"
done

t1=$(date +%s)
echo ""
echo "Total time: $((t1 - t0)) seconds"
echo "Prebuilt rnwhisper core libraries are in android/src/main/jniLibs/:"
ls -lh "$JNI_LIBS_DIR"/*/librnwhisper*.so
