# Troubleshooting

## Android: Got build error `Unknown host CPU architecture: arm64` on Apple Silicon Macs

To fix this, we recommended to use NDK version `24.0.8215888` or above.

If you're not able to change the root project for some reason, you can try to add `arch -x86_64 /bin/bash` to ndk-build script, for example, edit `~/Library/Android/sdk/ndk/23.1.7779620/ndk-build`:

```bash
#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
arch -x86_64 /bin/bash $DIR/build/ndk-build "$@"
```

## Android: `no prebuilt libraries for <abi>` / iOS: `ios/rnwhisper.xcframework is missing`

`whisper.rn` downloads its pre-built native libraries from the matching GitHub release during `npm install` (`postinstall`). If that step was skipped (offline install, `--ignore-scripts`, `RNWHISPER_SKIP_POSTINSTALL=1`, or a package manager that does not run dependency lifecycle scripts), either run it again:

```sh
npx whisper-rn-download-artifacts
```

or build from source instead: set `rnwhisperBuildFromSource=true` in `android/gradle.properties` and `RNWHISPER_BUILD_FROM_SOURCE=1` in your Podfile environment.

If the Android build reports an ABI you do not ship (e.g. `x86`), restrict `reactNativeArchitectures` in your `android/gradle.properties`.
