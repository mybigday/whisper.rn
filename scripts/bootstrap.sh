#!/bin/bash -e

git submodule init
git submodule update --recursive

cp ./whisper.cpp/ggml/include/ggml.h ./cpp/ggml.h
cp ./whisper.cpp/ggml/include/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./whisper.cpp/ggml/include/ggml-backend.h ./cpp/ggml-backend.h
cp ./whisper.cpp/ggml/include/ggml-metal.h ./cpp/ggml-metal.h

cp ./whisper.cpp/ggml/src/ggml.c ./cpp/ggml.c
cp ./whisper.cpp/ggml/src/ggml-metal.m ./cpp/ggml-metal.m
cp ./whisper.cpp/ggml/src/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./whisper.cpp/ggml/src/ggml-backend.cpp ./cpp/ggml-backend.cpp
cp ./whisper.cpp/ggml/src/ggml-backend-impl.h ./cpp/ggml-backend-impl.h
cp ./whisper.cpp/ggml/src/ggml-impl.h ./cpp/ggml-impl.h
cp ./whisper.cpp/ggml/src/ggml-cpu-impl.h ./cpp/ggml-cpu-impl.h
cp ./whisper.cpp/ggml/src/ggml-common.h ./cpp/ggml-common.h
cp ./whisper.cpp/ggml/src/ggml-quants.h ./cpp/ggml-quants.h
cp ./whisper.cpp/ggml/src/ggml-quants.c ./cpp/ggml-quants.c
cp ./whisper.cpp/ggml/src/ggml-aarch64.c ./cpp/ggml-aarch64.c
cp ./whisper.cpp/ggml/src/ggml-aarch64.h ./cpp/ggml-aarch64.h

cp ./whisper.cpp/include/whisper.h ./cpp/whisper.h
cp ./whisper.cpp/src/whisper.cpp ./cpp/whisper.cpp

rm -rf ./cpp/coreml/
cp -R ./whisper.cpp/src/coreml/ ./cpp/coreml/

# List of files to process
files=(
  "./cpp/ggml.h"
  "./cpp/ggml.c"
  "./cpp/ggml-metal.h"
  "./cpp/ggml-metal.m"
  "./cpp/ggml-quants.h"
  "./cpp/ggml-quants.c"
  "./cpp/ggml-alloc.h"
  "./cpp/ggml-alloc.c"
  "./cpp/ggml-backend.h"
  "./cpp/ggml-backend.cpp"
  "./cpp/ggml-backend-impl.h"
  "./cpp/ggml-impl.h"
  "./cpp/ggml-cpu-impl.h"
  "./cpp/ggml-common.h"
  "./cpp/ggml-aarch64.h"
  "./cpp/ggml-aarch64.c"
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

echo "Replacement completed successfully!"

# Parse whisper.cpp/bindings/javascript/package.json version and set to src/version.json
cd whisper.cpp/bindings/javascript
node -e "const fs = require('fs'); const package = JSON.parse(fs.readFileSync('package.json')); fs.writeFileSync('../../../src/version.json', JSON.stringify({version: package.version}));"
cd ../../../

yarn example

# Apply patch
patch -p0 -d ./cpp < ./scripts/ggml-backend.cpp.patch
patch -p0 -d ./cpp < ./scripts/ggml-metal.m.patch
# patch -p0 -d ./cpp < ./scripts/whisper.h.patch
patch -p0 -d ./cpp < ./scripts/whisper.cpp.patch

if [ "$OS" = "Darwin" ]; then
  # Build metallib (~1.4MB)
  cd whisper.cpp/ggml/src/
  xcrun --sdk iphoneos metal -c ggml-metal.metal -o ggml-metal.air
  xcrun --sdk iphoneos metallib ggml-metal.air   -o ggml-whisper.metallib
  rm ggml-metal.air
  cp ./ggml-whisper.metallib ../../../cpp/ggml-whisper.metallib

  cd -

  # Generate .xcode.env.local in iOS example
  cd example/ios
  echo export NODE_BINARY=$(command -v node) > .xcode.env.local

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
