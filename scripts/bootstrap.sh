#!/bin/bash -e
#
# Developer environment setup:
#   - example app dependencies
#   - the model and audio assets the example app bundles (example/assets/)
#
# The whisper.cpp sources under vendor/ are committed as-is (vendored +
# patched by scripts/sync-vendor.sh); this script never modifies them, so
# every build compiles exactly the checked-in tree.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_CPP_DIR="$ROOT_DIR/vendor/whisper.cpp"
ASSETS_DIR="$ROOT_DIR/example/assets"

cd "$ROOT_DIR"

yarn example

download() {
  local url="$1" dest="$2"
  if [ -x "$(command -v wget)" ]; then
    wget --no-config --quiet --show-progress -O "$dest" "$url"
  elif [ -x "$(command -v curl)" ]; then
    curl -L --output "$dest" "$url"
  else
    printf "Either wget or curl is required to download models.\n"
    exit 1
  fi
}

cp "$WHISPER_CPP_DIR/samples/jfk.wav" "$ASSETS_DIR/"

DUMMY_WHISPER_MODEL="$WHISPER_CPP_DIR/models/for-tests-ggml-base.bin"
DUMMY_VAD_MODEL="$WHISPER_CPP_DIR/models/for-tests-silero-v6.2.0-ggml.bin"
WHISPER_MODEL="$ASSETS_DIR/ggml-base.bin"
VAD_MODEL="$ASSETS_DIR/ggml-silero-v6.2.0.bin"

# If CI env is `true`, use dummy models
if [ "$CI" = "true" ]; then
  cp "$DUMMY_WHISPER_MODEL" "$WHISPER_MODEL"
  cp "$DUMMY_VAD_MODEL" "$VAD_MODEL"
  echo "CI: Copied dummy models to example/assets"
else
  # A previous CI bootstrap may have left the dummy models in place; replace
  # them with real ones.
  if cmp -s "$WHISPER_MODEL" "$DUMMY_WHISPER_MODEL"; then
    rm "$WHISPER_MODEL"
  fi
  if cmp -s "$VAD_MODEL" "$DUMMY_VAD_MODEL"; then
    rm "$VAD_MODEL"
  fi
  if [ ! -f "$WHISPER_MODEL" ]; then
    download https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin "$WHISPER_MODEL"
  fi
  if [ ! -f "$VAD_MODEL" ]; then
    download https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin "$VAD_MODEL"
  fi
fi

# Core ML encoder for the base model
if [ ! -d "$ASSETS_DIR/ggml-base-encoder.mlmodelc" ]; then
  ZIP="$ASSETS_DIR/ggml-base-encoder.mlmodelc.zip"
  download https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-encoder.mlmodelc.zip "$ZIP"
  unzip -q "$ZIP" -d "$ASSETS_DIR"
  rm "$ZIP"
fi
