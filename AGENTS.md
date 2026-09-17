# whisper.rn - Development Guidelines

## Project Overview

whisper.rn is a React Native binding for [whisper.cpp](https://github.com/ggerganov/whisper.cpp), enabling high-performance inference of OpenAI's Whisper automatic speech recognition (ASR) model on iOS and Android devices.

**Key Features:**
- Native speech-to-text transcription via whisper.cpp
- NVIDIA Parakeet TDT transcription with multilingual model support
- Voice Activity Detection (VAD) using Silero VAD model
- Realtime transcription with auto-slicing and memory management
- Core ML support for iOS (encoder acceleration)
- Metal/GPU acceleration support
- JSI (JavaScript Interface) bindings for efficient ArrayBuffer transfers

## General

- Pay attention to code readability.
- Add comments appropriately, no need to explain the obvious.
- Apply first-principles thinking when appropriate.

## Development Commands

### Setup and Bootstrap
```bash
yarn                    # Install dependencies
yarn bootstrap          # Setup example app deps + download example model assets
yarn sync:vendor        # Re-vendor vendor/whisper.cpp from vendor/VERSIONS + scripts/patches
```

### Code Quality
```bash
yarn typecheck          # Type-check TypeScript files
yarn lint               # Lint files with ESLint
yarn lint --fix         # Fix lint errors
yarn test               # Run Jest unit tests
```

### Build Commands
```bash
yarn build              # Build TypeScript library with react-native-builder-bob

# iOS framework builds (creates prebuilt xcframework)
yarn build:ios-frameworks    # Build rnwhisper.xcframework for iOS/tvOS
./scripts/build-ios.sh       # Manual iOS framework build script

# Example app builds
yarn example start      # Start Metro bundler for example app
yarn example android    # Run example on Android
yarn example ios        # Run example on iOS
yarn example pods       # Install iOS pods for example

# Release builds for testing performance
yarn example ios --mode Release
yarn example android --mode release
```

### Documentation
```bash
yarn docgen             # Generate API documentation with typedoc
```

### Release
```bash
yarn release            # Publish new version with release-it
```

## Architecture

### Multi-Layer Architecture

whisper.rn has a 4-layer architecture:

1. **JavaScript/TypeScript Layer** (`src/`):
   - `index.ts`: Main API exports, `WhisperContext`, `ParakeetContext`, and `WhisperVadContext` classes
   - `NativeRNWhisper.ts`: TurboModule spec (New Architecture compatible)
   - `realtime-transcription/`: Enhanced realtime transcription framework
     - `RealtimeTranscriber.ts`: Main transcriber with VAD integration
     - `SliceManager.ts`: Audio slice memory management
     - `adapters/`: Audio stream adapters (e.g., AudioPcmStreamAdapter)

2. **JSI Bindings** (`cpp/jsi/`):
   - `RNWhisperJSI.cpp/h`: High-performance JSI functions for ArrayBuffer operations
   - Direct memory transfer for audio data (bypasses JSON serialization)
   - Thread pool for async processing
   - Functions: `whisperTranscribeData`, `parakeetTranscribeData`, `whisperVadDetectSpeech`

3. **Native Modules** (`ios/`, `android/src/main/java/`):
   - **iOS**: `RNWhisper.mm`, `RNWhisperContext.mm`, `RNWhisperVadContext.mm`
   - **Android**: `RNWhisper.java`, `WhisperContext.java`, `WhisperVadContext.java`
   - React Native bridge methods for initialization, file transcription
   - Audio session management (iOS)
   - Realtime recording (deprecated, use RealtimeTranscriber instead)

4. **C++ Core** (`cpp/` + `vendor/`):
   - `cpp/` holds only whisper.rn's own code: `rn-whisper.cpp/h` (job management, transcription orchestration), `rn-whisper-log.h`, and `jsi/`
   - whisper.cpp is vendored under `vendor/whisper.cpp/` in its upstream layout (`include/`, `src/` with `whisper.cpp`, `parakeet.cpp` and `coreml/`, `ggml/{include,src}` with the CPU and Metal backends), pinned by `vendor/VERSIONS`, with whisper.rn changes kept as `-p1` patches in `scripts/patches/whisper.cpp/`. No git submodule, no symbol renaming. See `vendor/README.md`.
   - `vendor/whisper.cpp` is also consumed by whisper.node via upstream's CMake project, so it carries upstream's build files, `examples/common-whisper.*` (with `miniaudio.h`/`stb_vorbis.c`) and the BLAS/CUDA/Vulkan/WebGPU backends. whisper.rn's builds never compile those; `cmake/rnwhisper-sources.cmake` and `whisper-rn.podspec` filter them out (keep the two in sync).
   - `cmake/rnwhisper-sources.cmake` is the single source/include list every CMake build uses; `whisper-rn.podspec` mirrors it for CocoaPods

### Context Management

The library uses a context-based model:

- **WhisperContext**: Main transcription context, initialized with a GGML model file
  - Methods: `transcribe()`, `transcribeData()`, `transcribeRealtime()` (deprecated), `bench()`, `release()`
  - Supports file paths, base64 WAV, ArrayBuffer (via JSI), and asset URIs

- **ParakeetContext**: Parakeet TDT transcription context, initialized with a Parakeet GGUF model
  - Methods: `transcribe()`, `transcribeData()`, `release()`
  - Supports WAV file paths, base64 WAV, raw PCM16 ArrayBuffer data, and asset URIs

- **WhisperVadContext**: VAD context, initialized with Silero VAD model
  - Methods: `detectSpeech()`, `detectSpeechData()`, `release()`
  - Used for voice activity detection in audio segments

All contexts maintain:
- `id`: Unique context identifier
- `gpu`: Whether GPU/Metal acceleration is active
- `reasonNoGPU`: Explanation if GPU is not available

### JSI Installation

JSI bindings are lazily installed on first use:
- Called automatically by `initWhisper()`, `initParakeet()`, and `initWhisperVad()`
- Installs global functions including `whisperTranscribeData`, `parakeetTranscribeData`, and `whisperVadDetectSpeech`
- These functions are then captured and removed from global scope

### Realtime Transcription Architecture

The modern `RealtimeTranscriber` (in `src/realtime-transcription/`) provides:

1. **SliceManager**: Manages audio slicing with circular buffer strategy
   - Auto-slices at configurable duration (default: 30s)
   - Keeps limited slices in memory (default: 3)
   - Provides memory usage stats

2. **VAD Integration**:
   - Detects speech vs silence in audio slices
   - Triggers auto-slice on speech_end events
   - Configurable thresholds and presets

3. **Queue-based Processing**:
   - Transcription queue for sequential processing
   - One transcription at a time
   - Results stored by slice index

4. **Prompt Chaining**:
   - Optional initial prompt
   - Can chain previous slice results as context
   - Improves continuity in long transcriptions

5. **Audio Stream Adapters**:
   - Interface-based design for different audio sources
   - Built-in: `AudioPcmStreamAdapter` (uses @fugood/react-native-audio-pcm-stream)
   - Custom adapters can be implemented

### iOS-Specific Features

**Pre-built Framework**:
- By default, uses `ios/rnwhisper.xcframework` (pre-built)
- Set `RNWHISPER_BUILD_FROM_SOURCE=1` in Podfile to build from source
- Framework includes Metal shaders (`.metallib`)

**Core ML Support**:
- Accelerates encoder on iOS 15.0+
- Model files: `.mlmodelc` directories (model.mil, coremldata.bin, weights/weight.bin)
- Must be co-located with GGML model (e.g., `ggml-tiny.en.bin` → `ggml-tiny.en-encoder.mlmodelc/`)
- Set `useCoreMLIos: false` to disable even if files exist

**Build Flags**:
- `RNWHISPER_DISABLE_COREML=1`: Disable Core ML compilation
- `RNWHISPER_DISABLE_METAL=1`: Disable Metal GPU acceleration

### Android-Specific Features

**JNI Bridge**: `android/src/main/jni.cpp` connects Java to C++ whisper core

**Build Configuration**:
- NDK version 24.0.8215888+ recommended for Apple Silicon Macs
- Supports 16KB page sizes (Android 15+)
- Proguard rule required: `-keep class com.rnwhisper.** { *; }`

**CMake Build**: `android/CMakeLists.txt` controls native compilation

## File Patterns and Conventions

### Code Organization
- **TypeScript**: `src/` (library), `example/src/` (example app)
- **Native iOS**: `ios/*.{h,m,mm}` (Objective-C/C++)
- **Native Android**: `android/src/main/java/com/rnwhisper/*.java`
- **C++ Core**: `cpp/*.{cpp,h}`
- **JSI Bindings**: `cpp/jsi/*.{cpp,h}`
- **Tests**: `src/**/__tests__/*.test.ts`

### Naming Conventions
- Classes: PascalCase (`WhisperContext`, `RealtimeTranscriber`)
- Files: kebab-case for configs, PascalCase for classes
- Native modules: RNWhisper prefix (iOS/Android)
- C++ namespace: `rnwhisper`
- ggml/whisper.cpp symbols keep their upstream names (`ggml_*`, `whisper_*`); the vendored sources are not renamed. Coexistence with other ggml-based libraries (e.g. llama.rn) relies on each library being its own dynamic image (two-level namespace on Apple, `RTLD_LOCAL` plus `-Bsymbolic` on Android). The Objective-C `GGMLMetalClass` is process-global, so it is renamed per library with `-DGGMLMetalClass=RNWhisperGGMLMetalClass`.

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new features
- `fix:` bug fixes
- `refactor:` code refactoring
- `docs:` documentation
- `test:` tests
- `chore:` tooling/build changes

## Important Implementation Details

### Audio Format Requirements
- Sample rate: 16kHz (whisper requirement)
- Channels: Mono (1 channel)
- Format: 16-bit PCM
- Input: WAV files, base64 WAV, or raw PCM data (base64 or ArrayBuffer)

### Memory Management
- Contexts must be explicitly released: `context.release()` or `releaseAllWhisper()`
- Realtime transcription: Use `maxSlicesInMemory` to control buffer size
- Large models (medium/large) on iOS: Enable Extended Virtual Addressing entitlement

### Performance Optimization
- Use quantized models (q8, q5) to reduce size and improve speed on mobile
- Default thread count: 2 for 4-core devices, 4 for more cores
- Test in Release mode for accurate performance measurement
- GPU/Metal acceleration: Set `useGpu: true` (enabled by default)

### Asset Handling
- Use `require()` for bundled models/audio
- Add `.bin` and `.mil` to Metro config `assetExts`
- Max file size: 2GB (RN packager limitation)
- Alternative: Download models at runtime

### Deprecated APIs
- `transcribeRealtime()`: Use `RealtimeTranscriber` instead
  - Old API still works but lacks VAD auto-slicing and better memory management

## Platform-Specific Notes

### iOS
- Minimum iOS: 11.0, tvOS: 11.0
- Metal acceleration: iOS/tvOS only
- Audio Session management: Use `AudioSessionIos` utilities or `audioSessionOnStartIos`/`audioSessionOnStopIos` options
- Microphone permission: Add `NSMicrophoneUsageDescription` to Info.plist (tvOS: microphone not supported)

### Android
- Microphone permission: `RECORD_AUDIO` in AndroidManifest.xml
- Use `PermissionsAndroid` to request runtime permission
- Supports both Old and New Architecture

### Expo
- Requires prebuild: `npx expo prebuild`
- Follow standard React Native library integration

## Testing

### Unit Tests
- Jest with React Native preset
- Mock available: `jest.mock('whisper.rn', () => require('whisper.rn/jest-mock'))`
- Run: `yarn test`

### Example App
- Located in `example/`
- Uses tiny.en model and jfk.wav sample
- Demonstrates: Whisper and Parakeet transcription, VAD, realtime transcription
- Test realtime: Requires microphone permissions

## Build System

### Vendored sources (`scripts/sync-vendor.sh`)

1. Clones/fetches the whisper.cpp commit pinned in `vendor/VERSIONS` into `~/.cache/whisper.rn/`
2. Exports the subset whisper.rn builds into `vendor/whisper.cpp/`, unchanged and in upstream layout
3. Applies `scripts/patches/whisper.cpp/*.patch`
4. Regenerates `src/version.json` (whisper/ggml versions + commit; the builds pass them as `WHISPER_VERSION` / `PARAKEET_VERSION` / `GGML_VERSION` / `GGML_COMMIT` compile definitions)

Its output is committed. Run it after changing `vendor/VERSIONS` or a patch; running it on a clean tree must produce no diff.

### Patching whisper.cpp

1. Edit the vendored file in place, e.g. `vendor/whisper.cpp/src/whisper.cpp`
2. Regenerate its patch: `scripts/update-patch.sh src/whisper.cpp` writes `scripts/patches/whisper.cpp/whisper.cpp.patch` (optional second argument names the patch; a file upstream lacks becomes a file-creating patch; text above the first `---` line survives regeneration, use it to say why)
3. `yarn sync:vendor` must leave `git status` clean

### Bootstrap (`scripts/bootstrap.sh`)

Developer environment only; it never touches `vendor/`:

1. `example/` dependencies
2. Example model/audio assets into `example/assets/` (dummy models from `vendor/whisper.cpp/models/` when `CI=true`)

### iOS Framework Build Process
1. CMake generates Xcode project from `ios/CMakeLists.txt`
2. Builds for multiple targets: iOS device, iOS simulator, tvOS device, tvOS simulator
3. Creates universal `rnwhisper.xcframework` with all architectures
4. Includes Metal kernel sources (compiled at runtime) and the public C headers
5. Script: `scripts/build-ios.sh`

### TypeScript Build
- Uses `react-native-builder-bob`
- Outputs: CommonJS, ES Module, TypeScript definitions
- Output dir: `lib/`

### Android Build
- Gradle + CMake
- NDK builds JNI library
- Output: `librnwhisper.so`

## Key Dependencies

### Runtime
- `react-native`: Core framework
- `whisper.cpp`: C++ ASR engine (vendored at `vendor/whisper.cpp/`, see `vendor/README.md`)
- `safe-buffer`: Buffer polyfill

### Realtime Transcription
- `@fugood/react-native-audio-pcm-stream`: For `AudioPcmStreamAdapter`
- Filesystem module (e.g., `react-native-fs`): For WAV file writing

### Development
- `react-native-builder-bob`: TypeScript build
- `typedoc` + `typedoc-plugin-markdown`: API docs
- `@commitlint`, `lefthook`: Git hooks
- `release-it`: Publishing

## Troubleshooting Common Issues

### iOS Build Issues
- Clean derived data: `rm -rf ~/Library/Developer/Xcode/DerivedData`
- Clean build: `yarn clean && cd example/ios && pod install`
- Large models: Enable Extended Virtual Addressing entitlement

### Android Build Issues
- Unknown host CPU on Apple Silicon: Use NDK 24.0.8215888+
- See `docs/TROUBLESHOOTING.md` for more details

### Performance Issues
- Use Release mode for testing
- Choose appropriate model size for device
- Adjust `maxThreads` if needed
- Consider quantized models

## Related Documentation

- API Docs: `docs/API/`
- Tips & Tricks: `docs/TIPS.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- Contributing: `CONTRIBUTING.md`
- whisper.cpp upstream: https://github.com/ggerganov/whisper.cpp
