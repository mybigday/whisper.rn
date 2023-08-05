# Troubleshooting

## Android: Got build error `Unknown host CPU architecture: arm64` on Apple Silicon Macs

To fix this, we recommended to use NDK version `24.0.8215888` or above.

If you're not able to change the root project for some reason, you can try to add `arch -x86_64 /bin/bash` to ndk-build script, for example, edit `~/Library/Android/sdk/ndk/23.1.7779620/ndk-build`:

```bash
#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
arch -x86_64 /bin/bash $DIR/build/ndk-build "$@"
```
