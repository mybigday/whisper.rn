#!/bin/bash -e

git submodule init
git submodule update --recursive

cp ./whisper.cpp/ggml.h ./cpp/ggml.h
cp ./whisper.cpp/ggml.c ./cpp/ggml.c
cp ./whisper.cpp/whisper.h ./cpp/whisper.h
cp ./whisper.cpp/whisper.cpp ./cpp/whisper.cpp
cp -R ./whisper.cpp/coreml/ ./cpp/coreml/

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

# Copy to Android example
cp ggml-tiny.en.bin ../../example/android/app/src/main/assets
echo "Copied ggml-tiny.en.bin to example/android/app/src/main/assets"

# Check whisper.cpp/models/ggml-tiny.en-encoder.mlmodelc exist
if [ ! -d ./ggml-tiny.en-encoder.mlmodelc ]; then
  mkdir ggml-tiny.en-encoder.mlmodelc
  echo "Created a dummy ggml-tiny.en-encoder.mlmodelc for testing."
  echo "Please follow https://github.com/ggerganov/whisper.cpp#core-ml-support for convert a real model."
fi
