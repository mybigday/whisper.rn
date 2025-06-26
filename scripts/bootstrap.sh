#!/bin/bash -e

git submodule init
git submodule update --recursive

# ggml api
cp ./whisper.cpp/ggml/include/ggml.h ./cpp/ggml.h
cp ./whisper.cpp/ggml/include/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./whisper.cpp/ggml/include/ggml-backend.h ./cpp/ggml-backend.h
cp ./whisper.cpp/ggml/include/ggml-cpu.h ./cpp/ggml-cpu.h
cp ./whisper.cpp/ggml/include/ggml-cpp.h ./cpp/ggml-cpp.h
cp ./whisper.cpp/ggml/include/ggml-opt.h ./cpp/ggml-opt.h
cp ./whisper.cpp/ggml/include/ggml-metal.h ./cpp/ggml-metal.h
cp ./whisper.cpp/ggml/include/gguf.h ./cpp/gguf.h

cp ./whisper.cpp/ggml/src/ggml-metal/ggml-metal.m ./cpp/ggml-metal.m
cp ./whisper.cpp/ggml/src/ggml-metal/ggml-metal-impl.h ./cpp/ggml-metal-impl.h

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
  "./cpp/ggml-metal.m"
  "./cpp/ggml-metal-impl.h"
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
patch -p0 -d ./cpp < ./scripts/patches/ggml-metal.m.patch
patch -p0 -d ./cpp < ./scripts/patches/ggml-quants.c.patch
patch -p0 -d ./cpp < ./scripts/patches/whisper.h.patch
patch -p0 -d ./cpp < ./scripts/patches/whisper.cpp.patch
rm -rf ./cpp/*.orig

if [ "$OS" = "Darwin" ]; then
  # Build metallib (~2.6MB)
  cd whisper.cpp/ggml/src/ggml-metal

  # Create a symbolic link to ggml-common.h in the current directory
  ln -sf ../ggml-common.h .

  xcrun --sdk iphoneos metal -c ggml-metal.metal -o ggml-metal.air -DGGML_METAL_USE_BF16=1
  xcrun --sdk iphoneos metallib ggml-metal.air   -o ggml-whisper.metallib
  rm ggml-metal.air
  mv ./ggml-whisper.metallib ../../../../cpp/ggml-whisper.metallib

  xcrun --sdk iphonesimulator metal -c ggml-metal.metal -o ggml-metal.air -DGGML_METAL_USE_BF16=1
  xcrun --sdk iphonesimulator metallib ggml-metal.air   -o ggml-whisper.metallib
  rm ggml-metal.air
  mv ./ggml-whisper.metallib ../../../../cpp/ggml-whisper-sim.metallib

  # Remove the symbolic link
  rm ggml-common.h

  cd -
fi

# Download model for example
cd whisper.cpp/models

# If CI env is `true`, use dummy model
if [ "$CI" = "true" ]; then
  cp for-tests-ggml-base.bin ggml-base.bin
  echo "CI: Copied for-tests-ggml-base.bin to ggml-base.bin"
else
  ./download-ggml-model.sh base
fi

# Copy to assets
cp ../samples/jfk.wav ../../example/assets
cp ggml-base.bin ../../example/assets
echo "Copied ggml-base.bin to example/assets"

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
