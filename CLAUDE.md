# whisper.rn - Development Guidelines

## Project Overview

whisper.rn is a React Native binding for [whisper.cpp](https://github.com/ggerganov/whisper.cpp), enabling high-performance inference of OpenAI's Whisper automatic speech recognition (ASR) model on iOS and Android devices.

**Key Features:**
- Native speech-to-text transcription via whisper.cpp
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
yarn bootstrap          # Setup project (install deps + build whisper.cpp submodule)
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
   - `index.ts`: Main API exports, `WhisperContext` and `WhisperVadContext` classes
   - `NativeRNWhisper.ts`: TurboModule spec (New Architecture compatible)
   - `realtime-transcription/`: Enhanced realtime transcription framework
     - `RealtimeTranscriber.ts`: Main transcriber with VAD integration
     - `SliceManager.ts`: Audio slice memory management
     - `adapters/`: Audio stream adapters (e.g., AudioPcmStreamAdapter)

2. **JSI Bindings** (`cpp/jsi/`):
   - `RNWhisperJSI.cpp/h`: High-performance JSI functions for ArrayBuffer operations
   - Direct memory transfer for audio data (bypasses JSON serialization)
   - Thread pool for async processing
   - Functions: `whisperTranscribeData`, `whisperVadDetectSpeech`

3. **Native Modules** (`ios/`, `android/src/main/java/`):
   - **iOS**: `RNWhisper.mm`, `RNWhisperContext.mm`, `RNWhisperVadContext.mm`
   - **Android**: `RNWhisper.java`, `WhisperContext.java`, `WhisperVadContext.java`
   - React Native bridge methods for initialization, file transcription
   - Audio session management (iOS)
   - Realtime recording (deprecated, use RealtimeTranscriber instead)

4. **C++ Core** (`cpp/`):
   - `rn-whisper.cpp/h`: Job management, transcription orchestration
   - `rn-audioutils.cpp/h`: Audio conversion utilities (WAV, PCM)
   - `whisper.cpp`: Core whisper.cpp library (git submodule)
   - `ggml*.cpp/h`: GGML tensor library files (from whisper.cpp)
   - `coreml/`: Core ML integration headers (iOS)

### Context Management

The library uses a context-based model:

- **WhisperContext**: Main transcription context, initialized with a GGML model file
  - Methods: `transcribe()`, `transcribeData()`, `transcribeRealtime()` (deprecated), `bench()`, `release()`
  - Supports file paths, base64 WAV, ArrayBuffer (via JSI), and asset URIs

- **WhisperVadContext**: VAD context, initialized with Silero VAD model
  - Methods: `detectSpeech()`, `detectSpeechData()`, `release()`
  - Used for voice activity detection in audio segments

Both contexts maintain:
- `id`: Unique context identifier
- `gpu`: Whether GPU/Metal acceleration is active
- `reasonNoGPU`: Explanation if GPU is not available

### JSI Installation

JSI bindings are lazily installed on first use:
- Called automatically by `initWhisper()` and `initWhisperVad()`
- Installs global functions: `whisperTranscribeData`, `whisperVadDetectSpeech`
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
- Demonstrates: basic transcription, VAD, realtime transcription
- Test realtime: Requires microphone permissions

## Build System

### iOS Framework Build Process
1. CMake generates Xcode project from `ios/CMakeLists.txt`
2. Builds for multiple targets: iOS device, iOS simulator, tvOS device, tvOS simulator
3. Creates universal `rnwhisper.xcframework` with all architectures
4. Includes Metal shaders and C++ headers
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
- `whisper.cpp`: C++ ASR engine (submodule at `whisper.cpp/`)
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
