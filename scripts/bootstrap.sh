#!/bin/bash -e

git submodule init
git submodule update --recursive

cp ./whisper.cpp/ggml.h ./cpp/ggml.h
cp ./whisper.cpp/ggml.c ./cpp/ggml.c
cp ./whisper.cpp/whisper.h ./cpp/whisper.h
cp ./whisper.cpp/whisper.cpp ./cpp/whisper.cpp
cp -R ./whisper.cpp/coreml/ ./cpp/coreml/

yarn example

# Download model for example
cd whisper.cpp/models

# If CI env is `true`, use dummy model
if [ "$CI" = "true" ]; then
  cp for-tests-ggml-base.en.bin ggml-base.en.bin
  echo "CI: Copied for-tests-ggml-base.en.bin to ggml-base.en.bin"
else
  ./download-ggml-model.sh base.en
fi

# Copy to Android example
cp ggml-base.en.bin ../../example/android/app/src/main/assets
echo "Copied ggml-base.en.bin to example/android/app/src/main/assets"

# Check whisper.cpp/models/ggml-base.en-encoder.mlmodelc exist
if [ ! -d ./ggml-base.en-encoder.mlmodelc ]; then
  mkdir ggml-base.en-encoder.mlmodelc
  echo "Created a dummy ggml-base.en-encoder.mlmodelc for testing."
  echo "Please follow https://github.com/ggerganov/whisper.cpp#core-ml-support for convert a real model."
fi
