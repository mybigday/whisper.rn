# whisper.rn C++ sources

Only whisper.rn's own native code lives here. whisper.cpp (including ggml and
the Core ML glue) is vendored under `vendor/whisper.cpp` in its upstream
layout (see `vendor/README.md`), and the builds compile it from there.

- `rn-whisper.*`: context wrappers, job management and transcription
  orchestration on top of the whisper.cpp / Parakeet C APIs
- `rn-whisper-log.h`: platform logging macros
- `jsi/`: the JSI bindings that expose the above to JavaScript. Platform glue
  (`ios/RNWhisper.mm`, `android/src/main/jni.cpp`) installs them on app
  startup and provides model loading.

The source and include lists every CMake build uses are in
`cmake/rnwhisper-sources.cmake`; the CocoaPods equivalent is in
`whisper-rn.podspec`.

To change whisper.cpp behavior, edit the vendored file and regenerate its patch
with `scripts/update-patch.sh` (see `vendor/README.md`).
