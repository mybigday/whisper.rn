#!/bin/bash -e

git submodule init
git submodule update --recursive

cp ./whisper.cpp/ggml.h ./cpp/ggml.h
cp ./whisper.cpp/ggml.c ./cpp/ggml.c
cp ./whisper.cpp/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./whisper.cpp/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./whisper.cpp/ggml-metal.h ./cpp/ggml-metal.h
cp ./whisper.cpp/ggml-metal.m ./cpp/ggml-metal.m
cp ./whisper.cpp/ggml-metal.metal ./cpp/ggml-metal.metal
cp ./whisper.cpp/whisper.h ./cpp/whisper.h
cp ./whisper.cpp/whisper.cpp ./cpp/whisper.cpp

cp -R ./whisper.cpp/coreml/ ./cpp/coreml/

# List of files to process
files=(
  "./cpp/ggml.h"
  "./cpp/ggml.c"
  "./cpp/ggml-alloc.h"
  "./cpp/ggml-alloc.c"
  "./cpp/ggml-metal.h"
  "./cpp/ggml-metal.m"
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
  else
    sed -i 's/GGML_/WSP_GGML_/g' $file
    sed -i 's/ggml_/wsp_ggml_/g' $file
  fi
done

echo "Replacement completed successfully!"

# Parse whisper.cpp/bindings/javascript/package.json version and set to src/version.json
cd whisper.cpp/bindings/javascript
node -e "const fs = require('fs'); const package = JSON.parse(fs.readFileSync('package.json')); fs.writeFileSync('../../../src/version.json', JSON.stringify({version: package.version}));"
cd ../../../

yarn example

# Download model for example
cd whisper.cpp/models

# If CI env is `true`, use dummy model
if [ "$CI" = "true" ]; then
  cp for-tests-ggml-tiny.en.bin ggml-tiny.en.bin
  echo "CI: Copied for-tests-ggml-tiny.en.bin to ggml-tiny.en.bin"
else
  ./download-ggml-model.sh tiny.en
fi

# Copy to assets
cp ../samples/jfk.wav ../../example/assets
cp ggml-tiny.en.bin ../../example/assets
echo "Copied ggml-tiny.en.bin to example/assets"

# Check whisper.cpp/models/ggml-tiny.en-encoder.mlmodelc exist
if [ ! -d ./ggml-tiny.en-encoder.mlmodelc ]; then
  URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-encoder.mlmodelc.zip
  FILE=ggml-tiny.en-encoder.mlmodelc.zip

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

if [ ! -d ../../example/assets/ggml-tiny.en-encoder.mlmodelc ]; then
  cp -r ./ggml-tiny.en-encoder.mlmodelc ../../example/assets/
fi
