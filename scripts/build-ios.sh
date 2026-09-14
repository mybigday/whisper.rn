#!/bin/bash -e

if ! command -v cmake &> /dev/null; then
  echo "cmake could not be found, please install it"
  exit 1
fi

# Xcode generator is required for release builds: it produces frameworks with
# complete Info.plist metadata (MinimumOSVersion, CFBundleSupportedPlatforms,
# DT* keys). Frameworks built with "Unix Makefiles" lack these keys and cause
# App Store Connect to hang in TestFlight processing indefinitely.
if [ -n "$RNWHISPER_CMAKE_GENERATOR" ]; then
  CMAKE_GENERATOR=$RNWHISPER_CMAKE_GENERATOR
else
  CMAKE_GENERATOR="Xcode"
fi

# ggml/gguf keep upstream names; they must stay internal to this framework so a
# second ggml-based framework in the same app never resolves through it (see
# ios/unexported-symbols.txt).
function assert_no_ggml_exports() {
  local binary="$1"
  local leaked
  leaked="$(nm -gU "$binary" | awk '{print $3}' | grep -E '^_(ggml|gguf|quantize|dequantize|iq2xs|iq3xs)_|^__Z[A-Z]*[0-9]+(ggml|gguf)_|^__Z[A-Z]*N4ggml' || true)"
  if [ -n "$leaked" ]; then
    echo "ggml symbols exported from $binary:" >&2
    echo "$leaked" | head -20 >&2
    exit 1
  fi
}

function cp_headers() {
  mkdir -p ../ios/rnwhisper.xcframework/$1/rnwhisper.framework/Headers
  cp ../cpp/*.h ../ios/rnwhisper.xcframework/$1/rnwhisper.framework/Headers/
}

function build_framework() {
  # Parameters:
  # $1: system_name (iOS/tvOS)
  # $2: architectures
  # $3: sysroot
  # $4: output_path
  # $5: build_dir

  cd "$5"

  # Configure CMake
  cmake ../ios \
    -G"$CMAKE_GENERATOR" \
    -DCMAKE_SYSTEM_NAME=$1 \
    -DCMAKE_OSX_ARCHITECTURES="$2" \
    -DCMAKE_OSX_SYSROOT=$3 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=`pwd`/install \
    -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
    -DCMAKE_IOS_INSTALL_COMBINED=YES
  # Build
  cmake --build . --config Release -j $(sysctl -n hw.logicalcpu)

  # Setup framework directory
  rm -rf ../ios/rnwhisper.xcframework/$4
  mkdir -p ../ios/rnwhisper.xcframework/$4
  framework_path="Release-$3/rnwhisper.framework"
  if [ ! -d "$framework_path" ]; then
    framework_path="rnwhisper.framework"
  fi
  assert_no_ggml_exports "$framework_path/rnwhisper"
  mv "$framework_path" ../ios/rnwhisper.xcframework/$4/rnwhisper.framework
  mkdir -p ../ios/rnwhisper.xcframework/$4/rnwhisper.framework/Headers

  # Copy headers and split Metal kernel sources used for runtime compilation.
  framework_dir="../ios/rnwhisper.xcframework/$4/rnwhisper.framework"
  cp_headers $4
  mkdir -p "$framework_dir/kernels"
  cp ../cpp/ggml-metal/kernels/* "$framework_dir/kernels/"
  cp ../cpp/ggml-metal/ggml-metal-impl.h ../cpp/ggml-common.h "$framework_dir/"
  codesign --force --sign - --timestamp=none "$framework_dir"

  rm -rf ./*
  cd ..
}


t0=$(date +%s)

rm -rf build-ios
mkdir -p build-ios

# Build iOS frameworks
build_framework "iOS" "arm64;x86_64" "iphonesimulator" "ios-arm64_x86_64-simulator" "build-ios"
build_framework "iOS" "arm64" "iphoneos" "ios-arm64" "build-ios"
rm -rf build-ios

rm -rf build-tvos
mkdir -p build-tvos

# Build tvOS frameworks
build_framework "tvOS" "arm64;x86_64" "appletvsimulator" "tvos-arm64_x86_64-simulator" "build-tvos"
build_framework "tvOS" "arm64" "appletvos" "tvos-arm64" "build-tvos"
rm -rf build-tvos

t1=$(date +%s)
echo "Total time: $((t1 - t0)) seconds"
