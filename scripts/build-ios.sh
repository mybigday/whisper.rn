#!/bin/bash

set -euo pipefail

if ! command -v cmake &> /dev/null; then
  echo "cmake could not be found, please install it"
  exit 1
fi

# The Xcode generator is required: it produces frameworks with complete
# Info.plist metadata (MinimumOSVersion, CFBundleSupportedPlatforms, DT* keys)
# and emits the dSYM bundles packaged below. Frameworks built with
# "Unix Makefiles" lack these keys and hang forever in TestFlight processing.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/ios/rnwhisper.xcframework"
BUILD_ROOT="$ROOT_DIR/.build-ios-xcframework"
STAGING_DIR="$BUILD_ROOT/staging"

cleanup() {
  rm -rf "$BUILD_ROOT"
}

trap cleanup EXIT

copy_headers() {
  local framework_path="$1"

  mkdir -p "$framework_path/Headers"
  cp "$ROOT_DIR"/cpp/*.h "$framework_path/Headers/"
}

copy_framework_support_files() {
  local framework_path="$1"

  copy_headers "$framework_path"

  # Split Metal kernel sources used for runtime compilation.
  mkdir -p "$framework_path/kernels"
  cp "$ROOT_DIR"/cpp/ggml-metal/kernels/* "$framework_path/kernels/"
  cp "$ROOT_DIR"/cpp/ggml-metal/ggml-metal-impl.h "$ROOT_DIR"/cpp/ggml-common.h "$framework_path/"
}

# ggml/gguf keep upstream names; they must stay internal to this framework so a
# second ggml-based framework in the same app never resolves through it (see
# ios/unexported-symbols.txt).
assert_no_ggml_exports() {
  local binary="$1"
  local leaked

  leaked="$(nm -gU "$binary" | awk '{print $3}' | grep -E '^_(ggml|gguf|quantize|dequantize|iq2xs|iq3xs)_|^__Z[A-Z]*[0-9]+(ggml|gguf)_|^__Z[A-Z]*N4ggml' || true)"
  if [[ -n "$leaked" ]]; then
    echo "ggml symbols exported from $binary:" >&2
    echo "$leaked" | head -20 >&2
    exit 1
  fi
}

assert_matching_dsym() {
  local framework_path="$1"
  local dsym_path="$2"
  local binary_uuids
  local dsym_uuids

  if [[ ! -d "$dsym_path" ]]; then
    echo "Missing dSYM bundle for $framework_path" >&2
    exit 1
  fi

  binary_uuids="$(dwarfdump --uuid "$framework_path/rnwhisper" | awk '{print $2 ":" $3}' | sort)"
  dsym_uuids="$(dwarfdump --uuid "$dsym_path/Contents/Resources/DWARF/rnwhisper" | awk '{print $2 ":" $3}' | sort)"

  if [[ "$binary_uuids" != "$dsym_uuids" ]]; then
    echo "dSYM UUID mismatch for $framework_path" >&2
    echo "Framework UUIDs:" >&2
    echo "$binary_uuids" >&2
    echo "dSYM UUIDs:" >&2
    echo "$dsym_uuids" >&2
    exit 1
  fi
}

build_framework_slice() {
  # Parameters:
  # $1: system_name (iOS/tvOS)
  # $2: architectures
  # $3: sysroot
  # $4: library_identifier
  # $5: build_dir_name
  local system_name="$1"
  local architectures="$2"
  local sysroot="$3"
  local library_identifier="$4"
  local build_dir="$BUILD_ROOT/$5"
  local release_dir="$build_dir/Release-$sysroot"
  local framework_path="$release_dir/rnwhisper.framework"
  local dsym_path="$release_dir/rnwhisper.framework.dSYM"
  local staged_dir="$STAGING_DIR/$library_identifier"

  rm -rf "$build_dir" "$staged_dir"
  mkdir -p "$build_dir" "$staged_dir"

  (
    cd "$build_dir"

    cmake "$ROOT_DIR/ios" \
      -GXcode \
      -DCMAKE_SYSTEM_NAME="$system_name" \
      -DCMAKE_OSX_ARCHITECTURES="$architectures" \
      -DCMAKE_OSX_SYSROOT="$sysroot" \
      -DCMAKE_INSTALL_PREFIX="$PWD/install" \
      -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
      -DCMAKE_IOS_INSTALL_COMBINED=YES

    cmake --build . --config Release -j "$(sysctl -n hw.logicalcpu)"
  )

  if [[ ! -d "$framework_path" ]]; then
    echo "Missing framework build output at $framework_path" >&2
    exit 1
  fi

  assert_matching_dsym "$framework_path" "$dsym_path"
  assert_no_ggml_exports "$framework_path/rnwhisper"

  ditto "$framework_path" "$staged_dir/rnwhisper.framework"
  copy_framework_support_files "$staged_dir/rnwhisper.framework"
  ditto "$dsym_path" "$staged_dir/rnwhisper.framework.dSYM"
}

validate_packaged_dsym() {
  local library_identifier="$1"
  local packaged_framework="$OUTPUT_DIR/$library_identifier/rnwhisper.framework"
  local packaged_dsym

  packaged_dsym="$(find "$OUTPUT_DIR/$library_identifier" -maxdepth 3 -type d -name 'rnwhisper.framework.dSYM' -print -quit)"

  if [[ -z "$packaged_dsym" ]]; then
    echo "Missing packaged dSYM for $library_identifier in $OUTPUT_DIR" >&2
    exit 1
  fi

  assert_matching_dsym "$packaged_framework" "$packaged_dsym"
}

t0=$(date +%s)

rm -rf "$OUTPUT_DIR"
mkdir -p "$STAGING_DIR"

build_framework_slice "iOS" "arm64;x86_64" "iphonesimulator" \
  "ios-arm64_x86_64-simulator" "ios-simulator"
build_framework_slice "iOS" "arm64" "iphoneos" \
  "ios-arm64" "ios-device"
build_framework_slice "tvOS" "arm64;x86_64" "appletvsimulator" \
  "tvos-arm64_x86_64-simulator" "tvos-simulator"
build_framework_slice "tvOS" "arm64" "appletvos" \
  "tvos-arm64" "tvos-device"

xcodebuild -create-xcframework \
  -framework "$STAGING_DIR/ios-arm64_x86_64-simulator/rnwhisper.framework" \
  -debug-symbols "$STAGING_DIR/ios-arm64_x86_64-simulator/rnwhisper.framework.dSYM" \
  -framework "$STAGING_DIR/ios-arm64/rnwhisper.framework" \
  -debug-symbols "$STAGING_DIR/ios-arm64/rnwhisper.framework.dSYM" \
  -framework "$STAGING_DIR/tvos-arm64_x86_64-simulator/rnwhisper.framework" \
  -debug-symbols "$STAGING_DIR/tvos-arm64_x86_64-simulator/rnwhisper.framework.dSYM" \
  -framework "$STAGING_DIR/tvos-arm64/rnwhisper.framework" \
  -debug-symbols "$STAGING_DIR/tvos-arm64/rnwhisper.framework.dSYM" \
  -output "$OUTPUT_DIR"

validate_packaged_dsym "ios-arm64_x86_64-simulator"
validate_packaged_dsym "ios-arm64"
validate_packaged_dsym "tvos-arm64_x86_64-simulator"
validate_packaged_dsym "tvos-arm64"

t1=$(date +%s)
echo "Total time: $((t1 - t0)) seconds"
