# vendor

Vendored copies of whisper.cpp and vad.cpp, in their upstream directory layouts. There is no
git submodule: `git clone` gives you everything the builds need, and every
build (CocoaPods, the CMake projects) compiles these files in place.

| Directory | Upstream | What is vendored |
| --- | --- | --- |
| `whisper.cpp/` | [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | `include/`, `src/` (whisper, parakeet, `coreml/`), `ggml/{include,src}` (CPU, Metal, Hexagon, BLAS, CUDA, Vulkan and WebGPU backends), the audio-decoding helpers from `examples/`, the CMake project files, `LICENSE`, plus the dummy test models and `samples/jfk.wav` the example app uses |
| `vad.cpp/` | [mybigday/vad.cpp](https://github.com/mybigday/vad.cpp) | `include/vad.h`, `src/` (models, loaders, segmentation and the `vad-cpu` backend with its CPU kernels), `CMakeLists.txt`, `LICENSE` |

`VERSIONS` pins each upstream ref and resolved commit. The exact file lists
live in `scripts/sync-vendor.sh`.

vad.cpp builds against the ggml in `whisper.cpp/`; there is only one ggml.
Its `vad-cpu` backend includes ggml's internal backend headers
(`ggml-backend-impl.h`, `ggml-impl.h`), so the two pins move together: vad.cpp
pins its own `third_party/ggml` submodule to the ggml commit whisper.cpp
syncs, and a whisper.cpp update that moves ggml needs a matching vad.cpp.
`cmake/rnwhisper-sources.cmake` (`RNWHISPER_VAD_SOURCES`,
`rnwhisper_vad_configure_sources`) applies the same per-file flags as
vad.cpp's own CMake project. On Android arm64 that includes the ARMv8.2
kernels, which are selected at runtime.

`whisper.cpp/` serves two consumers. whisper.rn's builds pick their sources from
`cmake/rnwhisper-sources.cmake` and `whisper-rn.podspec`. [whisper.node](https://github.com/mybigday/whisper.node)
syncs whisper.rn and builds the same tree through upstream's own CMake project
(`add_subdirectory(vendor/whisper.cpp)`), which is why the tree also carries the
`CMakeLists.txt`/`cmake/` files, `examples/common-whisper.*` with `miniaudio.h`
and `stb_vorbis.c`, the BLAS/CUDA/Vulkan/WebGPU backends and the wasm CPU kernels.
whisper.rn never compiles those; its source lists filter them out, and the npm
package leaves out the CUDA, Vulkan and WebGPU backends. Note that upstream's
`cmake/build-info.cmake` reads the enclosing git checkout, so a build through
that project reports whisper.rn's commit, not whisper.cpp's; the pinned upstream
commit is in `VERSIONS` and `src/version.json`.

## Updating whisper.cpp or vad.cpp

1. Edit the `WHISPER_CPP_REF` or `VAD_CPP_REF` line in `VERSIONS` (a tag such as `v1.9.3`, a branch, or a commit).
2. Run `yarn sync:vendor`. It fetches the upstream repo into
   `~/.cache/whisper.rn/` (override with `WHISPER_RN_CACHE_DIR`), exports the
   subset, applies `scripts/patches/<name>/*.patch`, regenerates
   `src/version.json`, and writes the resolved `*_COMMIT` back.
3. Fix any patch that no longer applies (see below), then commit the result.

## Patching upstream code

Patches are plain `-p1` unified diffs against the upstream tree, one file per
patch, under `scripts/patches/whisper.cpp/` and `scripts/patches/vad.cpp/`. Text above the first `---` line
is kept when a patch is regenerated, so use it to explain why the patch exists.

To change or add one:

1. Edit the file in place, e.g. `vendor/whisper.cpp/src/whisper.cpp`.
2. Regenerate the patch from the pristine upstream file:

   ```bash
   scripts/update-patch.sh src/whisper.cpp
   ```

   The optional second argument names the patch file (default: the file's
   basename). A file upstream does not have becomes a file-creating patch.
   For vad.cpp, prefix the path: `scripts/update-patch.sh vad.cpp/src/vad.cpp`.
3. `yarn sync:vendor` must leave the tree unchanged (`git status` clean);
   that is the check that the patch set is complete.

Do not edit vendored files without regenerating the patch: the next sync
re-exports upstream and reapplies only what is in `scripts/patches/`.

## Version information

whisper.cpp passes `WHISPER_VERSION`, `PARAKEET_VERSION`, `GGML_VERSION` and
`GGML_COMMIT` as compile definitions from its own CMake project. The sync
records them in `src/version.json`; `cmake/rnwhisper-sources.cmake` and
`whisper-rn.podspec` read that file and define the same macros.
