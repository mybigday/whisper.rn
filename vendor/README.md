# vendor

Vendored copy of whisper.cpp, in its upstream directory layout. There is no
git submodule: `git clone` gives you everything the builds need, and every
build (CocoaPods, the CMake projects) compiles these files in place.

| Directory | Upstream | What is vendored |
| --- | --- | --- |
| `whisper.cpp/` | [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | `include/`, `src/` (whisper, parakeet, `coreml/`), `ggml/{include,src}` (CPU, Metal and Hexagon backends), `LICENSE`, plus the dummy test models and `samples/jfk.wav` the example app uses |

`VERSIONS` pins the upstream ref and resolved commit. The exact file list
lives in `scripts/sync-vendor.sh`.

## Updating whisper.cpp

1. Edit the `WHISPER_CPP_REF` line in `VERSIONS` (a tag such as `v1.9.3`, a branch, or a commit).
2. Run `yarn sync:vendor`. It fetches the upstream repo into
   `~/.cache/whisper.rn/` (override with `WHISPER_RN_CACHE_DIR`), exports the
   subset, applies `scripts/patches/whisper.cpp/*.patch`, regenerates
   `src/version.json`, and writes the resolved `WHISPER_CPP_COMMIT` back.
3. Fix any patch that no longer applies (see below), then commit the result.

## Patching upstream code

Patches are plain `-p1` unified diffs against the upstream tree, one file per
patch, under `scripts/patches/whisper.cpp/`. Text above the first `---` line
is kept when a patch is regenerated, so use it to explain why the patch exists.

To change or add one:

1. Edit the file in place, e.g. `vendor/whisper.cpp/src/whisper.cpp`.
2. Regenerate the patch from the pristine upstream file:

   ```bash
   scripts/update-patch.sh src/whisper.cpp
   ```

   The optional second argument names the patch file (default: the file's
   basename). A file upstream does not have becomes a file-creating patch.
3. `yarn sync:vendor` must leave the tree unchanged (`git status` clean);
   that is the check that the patch set is complete.

Do not edit vendored files without regenerating the patch: the next sync
re-exports upstream and reapplies only what is in `scripts/patches/`.

## Version information

whisper.cpp passes `WHISPER_VERSION`, `PARAKEET_VERSION`, `GGML_VERSION` and
`GGML_COMMIT` as compile definitions from its own CMake project. The sync
records them in `src/version.json`; `cmake/rnwhisper-sources.cmake` and
`whisper-rn.podspec` read that file and define the same macros.
