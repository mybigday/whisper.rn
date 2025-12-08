#!/bin/bash -e

ROOT_DIR=$(pwd)
OS=$(uname)

git submodule init
git submodule update --recursive

# Hexagon SDK setup for Android builds
echo ""
echo "=========================================="
echo "Hexagon SDK Setup"
echo "=========================================="
echo ""

# Check if Docker is available and recommend it
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
  echo "Docker is available!"
  echo ""
  echo "For Hexagon builds, we recommend using Docker for consistent builds."
  echo "Docker provides a pre-configured environment with all dependencies."
  echo ""
  echo "Build commands:"
  echo "  ./scripts/build-hexagon-htp.sh       - Build HTP libraries (auto-detects Docker)"
  echo ""

  # Pull Docker image in background
  DOCKER_IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.3"
  if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo "Pulling Docker image in background..."
    echo "  Image: $DOCKER_IMAGE"
    docker pull "$DOCKER_IMAGE" &
    DOCKER_PULL_PID=$!
    echo "  (Pull process running in background, PID: $DOCKER_PULL_PID)"
  else
    echo "Docker image already present: $DOCKER_IMAGE"
  fi
  echo ""
else
  echo "Docker not available. You can:"
  echo "  1. Install Docker for consistent builds (recommended)"
  echo "  2. Install Hexagon SDK manually for native Linux builds"
  echo ""
fi

# Download and setup Hexagon SDK (for all platforms)
# On macOS: Needed for libcdsprpc.so linking when building Android libraries
# On Linux: Can be used for native builds without Docker
HEXAGON_SDK_VERSION="6.4.0.2"
HEXAGON_TOOLS_VERSION="19.0.04"
HEXAGON_INSTALL_DIR="${HEXAGON_INSTALL_DIR:-$HOME/.hexagon-sdk}"

if [ ! -d "$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION" ]; then
  echo "Downloading Hexagon SDK v${HEXAGON_SDK_VERSION}..."
  echo ""

  if [ "$OS" = "Darwin" ]; then
    echo "Note: SDK tools won't run on macOS, but libcdsprpc.so is needed for linking"
  fi
  echo ""

  TEMP_DIR=$(mktemp -d)
  cd "$TEMP_DIR"

  curl -L -o hex-sdk.tar.gz \
    "https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v${HEXAGON_SDK_VERSION}/hexagon-sdk-v${HEXAGON_SDK_VERSION}-amd64-lnx.tar.xz"

  echo "Extracting Hexagon SDK..."
  mkdir -p "$HEXAGON_INSTALL_DIR"
  tar -xaf hex-sdk.tar.gz -C "$HEXAGON_INSTALL_DIR"

  cd "$ROOT_DIR"
  rm -rf "$TEMP_DIR"

  echo "Hexagon SDK installed to: $HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
  echo ""
  echo "The build scripts will automatically detect and use the SDK."
  echo ""
  echo "To build HTP libraries:"
  echo "  ./scripts/build-hexagon-htp.sh"
  echo ""
else
  echo "Hexagon SDK installed: $HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
  echo ""
fi

echo "=========================================="
echo ""

# ggml api
cp ./whisper.cpp/ggml/include/ggml.h ./cpp/ggml.h
cp ./whisper.cpp/ggml/include/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./whisper.cpp/ggml/include/ggml-backend.h ./cpp/ggml-backend.h
cp ./whisper.cpp/ggml/include/ggml-cpu.h ./cpp/ggml-cpu.h
cp ./whisper.cpp/ggml/include/ggml-cpp.h ./cpp/ggml-cpp.h
cp ./whisper.cpp/ggml/include/ggml-opt.h ./cpp/ggml-opt.h
cp ./whisper.cpp/ggml/include/ggml-metal.h ./cpp/ggml-metal.h
cp ./whisper.cpp/ggml/include/gguf.h ./cpp/gguf.h

cp -r ./whisper.cpp/ggml/src/ggml-metal ./cpp/
rm ./cpp/ggml-metal/CMakeLists.txt

# ggml-hexagon (Qualcomm Hexagon DSP backend)
cp ./whisper.cpp/ggml/include/ggml-hexagon.h ./cpp/ggml-hexagon.h
cp -r ./whisper.cpp/ggml/src/ggml-hexagon ./cpp/
# Keep CMakeLists.txt for hexagon as it's needed for building HTP components

# Embed headers into ggml-metal.metal for runtime compilation
# This allows the .metal file to be compiled at runtime without needing external header files
METAL_SOURCE="./cpp/ggml-metal/ggml-metal.metal"
METAL_TMP="./cpp/ggml-metal/ggml-metal.metal.tmp"
COMMON_HEADER="./whisper.cpp/ggml/src/ggml-common.h"
IMPL_HEADER="./cpp/ggml-metal/ggml-metal-impl.h"

# Step 1: Replace the __embed_ggml-common.h__ placeholder with the actual header content
awk '
/^#if defined\(GGML_METAL_EMBED_LIBRARY\)/ { skip=1; next }
/__embed_ggml-common.h__/ {
    system("cat '"$COMMON_HEADER"'")
    next
}
/^#else/ && skip { skip_else=1; next }
/^#endif/ && skip_else { skip=0; skip_else=0; next }
!skip { print }
' < "$METAL_SOURCE" > "$METAL_TMP"

# Step 2: Replace the #include "ggml-metal-impl.h" with the actual header content
sed -e '/#include "ggml-metal-impl.h"/r '"$IMPL_HEADER" \
    -e '/#include "ggml-metal-impl.h"/d' < "$METAL_TMP" > "$METAL_SOURCE"
rm "$METAL_TMP"

cp ./whisper.cpp/ggml/src/ggml-cpu/ggml-cpu.c ./cpp/ggml-cpu/ggml-cpu.c
cp ./whisper.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp ./cpp/ggml-cpu/ggml-cpu.cpp
cp ./whisper.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h ./cpp/ggml-cpu/ggml-cpu-impl.h
cp ./whisper.cpp/ggml/src/ggml-cpu/quants.h ./cpp/ggml-cpu/quants.h
cp ./whisper.cpp/ggml/src/ggml-cpu/quants.c ./cpp/ggml-cpu/quants.c
cp ./whisper.cpp/ggml/src/ggml-cpu/arch-fallback.h ./cpp/ggml-cpu/arch-fallback.h
cp ./whisper.cpp/ggml/src/ggml-cpu/repack.cpp ./cpp/ggml-cpu/repack.cpp
cp ./whisper.cpp/ggml/src/ggml-cpu/repack.h ./cpp/ggml-cpu/repack.h
cp ./whisper.cpp/ggml/src/ggml-cpu/traits.h ./cpp/ggml-cpu/traits.h
cp ./whisper.cpp/ggml/src/ggml-cpu/traits.cpp ./cpp/ggml-cpu/traits.cpp
cp ./whisper.cpp/ggml/src/ggml-cpu/common.h ./cpp/ggml-cpu/common.h

cp ./whisper.cpp/ggml/src/ggml-cpu/unary-ops.h ./cpp/ggml-cpu/unary-ops.h
cp ./whisper.cpp/ggml/src/ggml-cpu/unary-ops.cpp ./cpp/ggml-cpu/unary-ops.cpp
cp ./whisper.cpp/ggml/src/ggml-cpu/binary-ops.h ./cpp/ggml-cpu/binary-ops.h
cp ./whisper.cpp/ggml/src/ggml-cpu/binary-ops.cpp ./cpp/ggml-cpu/binary-ops.cpp
cp ./whisper.cpp/ggml/src/ggml-cpu/vec.h ./cpp/ggml-cpu/vec.h
cp ./whisper.cpp/ggml/src/ggml-cpu/vec.cpp ./cpp/ggml-cpu/vec.cpp
cp ./whisper.cpp/ggml/src/ggml-cpu/simd-mappings.h ./cpp/ggml-cpu/simd-mappings.h
cp ./whisper.cpp/ggml/src/ggml-cpu/ops.h ./cpp/ggml-cpu/ops.h
cp ./whisper.cpp/ggml/src/ggml-cpu/ops.cpp ./cpp/ggml-cpu/ops.cpp

cp -r ./whisper.cpp/ggml/src/ggml-cpu/amx ./cpp/ggml-cpu/
mkdir -p ./cpp/ggml-cpu/arch
cp -r ./whisper.cpp/ggml/src/ggml-cpu/arch/arm ./cpp/ggml-cpu/arch/
cp -r ./whisper.cpp/ggml/src/ggml-cpu/arch/x86 ./cpp/ggml-cpu/arch/

cp ./whisper.cpp/ggml/src/ggml.c ./cpp/ggml.c
cp ./whisper.cpp/ggml/src/ggml-impl.h ./cpp/ggml-impl.h
cp ./whisper.cpp/ggml/src/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./whisper.cpp/ggml/src/ggml-backend.cpp ./cpp/ggml-backend.cpp
cp ./whisper.cpp/ggml/src/ggml-backend-impl.h ./cpp/ggml-backend-impl.h
cp ./whisper.cpp/ggml/src/ggml-backend-reg.cpp ./cpp/ggml-backend-reg.cpp
cp ./whisper.cpp/ggml/src/ggml-common.h ./cpp/ggml-common.h
cp ./whisper.cpp/ggml/src/ggml-opt.cpp ./cpp/ggml-opt.cpp
cp ./whisper.cpp/ggml/src/ggml-quants.h ./cpp/ggml-quants.h
cp ./whisper.cpp/ggml/src/ggml-quants.c ./cpp/ggml-quants.c
cp ./whisper.cpp/ggml/src/ggml-threading.cpp ./cpp/ggml-threading.cpp
cp ./whisper.cpp/ggml/src/ggml-threading.h ./cpp/ggml-threading.h
cp ./whisper.cpp/ggml/src/gguf.cpp ./cpp/gguf.cpp

# whisper api
cp ./whisper.cpp/include/whisper.h ./cpp/whisper.h
cp ./whisper.cpp/src/whisper-arch.h ./cpp/whisper-arch.h
cp ./whisper.cpp/src/whisper.cpp ./cpp/whisper.cpp

rm -rf ./cpp/coreml/
cp -R ./whisper.cpp/src/coreml/ ./cpp/coreml/

# List of files to process
files=(
  # ggml api
  "./cpp/ggml-common.h"
  "./cpp/ggml.h"
  "./cpp/ggml.c"
  "./cpp/gguf.h"
  "./cpp/gguf.cpp"
  "./cpp/ggml-impl.h"
  "./cpp/ggml-cpp.h"
  "./cpp/ggml-opt.h"
  "./cpp/ggml-opt.cpp"
  "./cpp/ggml-metal.h"
  "./cpp/ggml-metal/ggml-metal.cpp"
  "./cpp/ggml-metal/ggml-metal-impl.h"
  "./cpp/ggml-metal/ggml-metal-common.h"
  "./cpp/ggml-metal/ggml-metal-common.cpp"
  "./cpp/ggml-metal/ggml-metal-context.h"
  "./cpp/ggml-metal/ggml-metal-context.m"
  "./cpp/ggml-metal/ggml-metal-device.h"
  "./cpp/ggml-metal/ggml-metal-device.cpp"
  "./cpp/ggml-metal/ggml-metal-device.m"
  "./cpp/ggml-metal/ggml-metal-ops.h"
  "./cpp/ggml-metal/ggml-metal-ops.cpp"
  "./cpp/ggml-metal/ggml-metal.metal"
  # ggml-hexagon (Qualcomm Hexagon DSP backend)
  "./cpp/ggml-hexagon.h"
  "./cpp/ggml-hexagon/ggml-hexagon.cpp"
  "./cpp/ggml-hexagon/htp-utils.c"
  "./cpp/ggml-hexagon/htp-utils.h"
  "./cpp/ggml-hexagon/htp/main.c"
  "./cpp/ggml-hexagon/htp/htp-ctx.h"
  "./cpp/ggml-hexagon/htp/htp-dma.c"
  "./cpp/ggml-hexagon/htp/htp-dma.h"
  "./cpp/ggml-hexagon/htp/htp-msg.h"
  "./cpp/ggml-hexagon/htp/htp-ops.h"
  "./cpp/ggml-hexagon/htp/worker-pool.c"
  "./cpp/ggml-hexagon/htp/worker-pool.h"
  "./cpp/ggml-hexagon/htp/matmul-ops.c"
  "./cpp/ggml-hexagon/htp/act-ops.c"
  "./cpp/ggml-hexagon/htp/binary-ops.c"
  "./cpp/ggml-hexagon/htp/unary-ops.c"
  "./cpp/ggml-hexagon/htp/rope-ops.c"
  "./cpp/ggml-hexagon/htp/softmax-ops.c"
  "./cpp/ggml-hexagon/htp/ops-utils.h"
  "./cpp/ggml-hexagon/htp/hvx-utils.c"
  "./cpp/ggml-hexagon/htp/hvx-utils.h"
  "./cpp/ggml-hexagon/htp/hvx-exp.c"
  "./cpp/ggml-hexagon/htp/hvx-inverse.c"
  "./cpp/ggml-hexagon/htp/hvx-sigmoid.c"
  "./cpp/ggml-quants.h"
  "./cpp/ggml-quants.c"
  "./cpp/ggml-alloc.h"
  "./cpp/ggml-alloc.c"
  "./cpp/ggml-backend.h"
  "./cpp/ggml-backend.cpp"
  "./cpp/ggml-backend-impl.h"
  "./cpp/ggml-backend-reg.cpp"
  "./cpp/ggml-cpu.h"
  "./cpp/ggml-cpu/ggml-cpu-impl.h"
  "./cpp/ggml-cpu/ggml-cpu.c"
  "./cpp/ggml-cpu/ggml-cpu.cpp"
  "./cpp/ggml-cpu/quants.h"
  "./cpp/ggml-cpu/quants.c"
  "./cpp/ggml-cpu/traits.h"
  "./cpp/ggml-cpu/traits.cpp"
  "./cpp/ggml-cpu/arch-fallback.h"
  "./cpp/ggml-cpu/repack.cpp"
  "./cpp/ggml-cpu/repack.h"
  "./cpp/ggml-cpu/common.h"
  "./cpp/ggml-threading.h"
  "./cpp/ggml-threading.cpp"
  "./cpp/ggml-cpu/amx/amx.h"
  "./cpp/ggml-cpu/amx/amx.cpp"
  "./cpp/ggml-cpu/amx/mmq.h"
  "./cpp/ggml-cpu/amx/mmq.cpp"
  "./cpp/ggml-cpu/amx/common.h"
  "./cpp/ggml-cpu/unary-ops.h"
  "./cpp/ggml-cpu/unary-ops.cpp"
  "./cpp/ggml-cpu/binary-ops.h"
  "./cpp/ggml-cpu/binary-ops.cpp"
  "./cpp/ggml-cpu/vec.h"
  "./cpp/ggml-cpu/vec.cpp"
  "./cpp/ggml-cpu/simd-mappings.h"
  "./cpp/ggml-cpu/ops.h"
  "./cpp/ggml-cpu/ops.cpp"
  "./cpp/ggml-cpu/arch/arm/cpu-feats.cpp"
  "./cpp/ggml-cpu/arch/arm/quants.c"
  "./cpp/ggml-cpu/arch/arm/repack.cpp"
  "./cpp/ggml-cpu/arch/x86/cpu-feats.cpp"
  "./cpp/ggml-cpu/arch/x86/quants.c"
  "./cpp/ggml-cpu/arch/x86/repack.cpp"

  # whisper api
  "./cpp/whisper-arch.h"
  "./cpp/whisper.h"
  "./cpp/whisper.cpp"
)

# Loop through each file and run the sed commands
OS=$(uname)
for file in "${files[@]}"; do
  # Add prefix to avoid redefinition with other libraries using ggml like llama.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/GGML_/WSP_GGML_/g' $file
    sed -i '' 's/ggml_/wsp_ggml_/g' $file
    sed -i '' 's/GGUF_/WSP_GGUF_/g' $file
    sed -i '' 's/gguf_/wsp_gguf_/g' $file
    sed -i '' 's/GGMLMetalClass/WSPGGMLMetalClass/g' $file
    sed -i '' 's/dequantize_/wsp_dequantize_/g' $file
    sed -i '' 's/quantize_/wsp_quantize_/g' $file
  else
    sed -i 's/GGML_/WSP_GGML_/g' $file
    sed -i 's/ggml_/wsp_ggml_/g' $file
    sed -i 's/GGUF_/WSP_GGUF_/g' $file
    sed -i 's/gguf_/wsp_gguf_/g' $file
    sed -i 's/GGMLMetalClass/WSPGGMLMetalClass/g' $file
    sed -i 's/dequantize_/wsp_dequantize_/g' $file
    sed -i 's/quantize_/wsp_quantize_/g' $file
  fi
done

files_iq_add_wsp_prefix=(
  "./cpp/ggml-quants.h"
  "./cpp/ggml-quants.c"
  "./cpp/ggml.c"
)

for file in "${files_iq_add_wsp_prefix[@]}"; do
  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/iq2xs_init_impl/wsp_iq2xs_init_impl/g' $file
    sed -i '' 's/iq2xs_free_impl/wsp_iq2xs_free_impl/g' $file
    sed -i '' 's/iq3xs_init_impl/wsp_iq3xs_init_impl/g' $file
    sed -i '' 's/iq3xs_free_impl/wsp_iq3xs_free_impl/g' $file
  else
    sed -i 's/iq2xs_init_impl/wsp_iq2xs_init_impl/g' $file
    sed -i 's/iq2xs_free_impl/wsp_iq2xs_free_impl/g' $file
    sed -i 's/iq3xs_init_impl/wsp_iq3xs_init_impl/g' $file
    sed -i 's/iq3xs_free_impl/wsp_iq3xs_free_impl/g' $file
  fi
done

echo "Replacement completed successfully!"

# Parse whisper.cpp/bindings/javascript/package.json version and set to src/version.json
cd whisper.cpp/bindings/javascript
node -e "const fs = require('fs'); const package = JSON.parse(fs.readFileSync('package.json')); fs.writeFileSync('../../../src/version.json', JSON.stringify({version: package.version}));"
cd ../../../

yarn example

# Apply patch
patch -p0 -d ./cpp < ./scripts/patches/ggml-quants.c.patch
patch -p0 -d ./cpp < ./scripts/patches/ggml.c.patch
patch -p0 -d ./cpp < ./scripts/patches/whisper.h.patch
patch -p0 -d ./cpp < ./scripts/patches/whisper.cpp.patch
patch -p0 -d ./cpp < ./scripts/patches/ggml-hexagon.cpp.patch
rm -rf ./cpp/*.orig
rm -rf ./cpp/**/*.orig

# Download model for example
cd whisper.cpp/models

# If CI env is `true`, use dummy model
if [ "$CI" = "true" ]; then
  cp for-tests-ggml-base.bin ggml-base.bin
  cp for-tests-silero-v5.1.2-ggml.bin ggml-silero-v5.1.2.bin
  echo "CI: Copied for-tests-ggml-base.bin to ggml-base.bin"
else
  ./download-ggml-model.sh base
  ./download-vad-model.sh silero-v5.1.2
fi

# Copy to assets
cp ../samples/jfk.wav ../../example/assets
cp ggml-base.bin ../../example/assets
echo "Copied ggml-base.bin to example/assets"
cp ggml-silero-v5.1.2.bin ../../example/assets
echo "Copied ggml-silero-v5.1.2.bin to example/assets"

# Check whisper.cpp/models/ggml-base-encoder.mlmodelc exist
if [ ! -d ./ggml-base-encoder.mlmodelc ]; then
  URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-encoder.mlmodelc.zip
  FILE=ggml-base-encoder.mlmodelc.zip

  if [ -x "$(command -v wget)" ]; then
    wget --no-config --quiet --show-progress -O $FILE $URL
  elif [ -x "$(command -v curl)" ]; then
    curl -L --output $FILE $URL
  else
    printf "Either wget or curl is required to download models.\n"
    exit 1
  fi

  unzip $FILE
  rm $FILE
fi

if [ ! -d ../../example/assets/ggml-base-encoder.mlmodelc ]; then
  cp -r ./ggml-base-encoder.mlmodelc ../../example/assets/
fi
