# whisper.rn

[![Actions Status](https://github.com/mybigday/whisper.rn/workflows/CI/badge.svg)](https://github.com/mybigday/whisper.rn/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/npm/v/whisper.rn.svg)](https://www.npmjs.com/package/whisper.rn/)

React Native bindings for Whisper and NVIDIA Parakeet ASR through [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

[whisper.cpp](https://github.com/ggerganov/whisper.cpp): High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model

## Screenshots

| <img src="https://github.com/mybigday/whisper.rn/assets/3001525/2fea7b2d-c911-44fb-9afc-8efc7b594446" width="300" /> | <img src="https://github.com/mybigday/whisper.rn/assets/3001525/a5005a6c-44f7-4db9-95e8-0fd951a2e147" width="300" /> |
| :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
|                                           iOS: Tested on iPhone 13 Pro Max                                           |                                              Android: Tested on Pixel 6                                              |
|                                  (tiny.en, Core ML enabled, release mode + archive)                                  |                                       (tiny.en, armv8.2-a+fp16, release mode)                                        |

## Installation

```sh
npm install whisper.rn
```

`whisper.rn` downloads the pre-built `ios/rnwhisper.xcframework` and `android/src/main/jniLibs` from the matching GitHub release during `postinstall`. Existing downloads are reused, and each archive is verified with SHA-256 before extraction. Set `RNWHISPER_SKIP_POSTINSTALL=1` to skip the download (e.g. when building from source), or run `npx whisper-rn-download-artifacts` to fetch it again.

#### iOS

Please re-run `npx pod-install` again.

By default, `whisper.rn` will use pre-built `rnwhisper.xcframework` for iOS. If you want to build from source, please set `RNWHISPER_BUILD_FROM_SOURCE` to `1` in your Podfile.

If you want to use `medium` or `large` model, the [Extended Virtual Addressing](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_extended-virtual-addressing) capability is recommended to enable on iOS project.

#### Android

Add proguard rule if it's enabled in project (android/app/proguard-rules.pro):

```proguard
# whisper.rn
-keep class com.rnwhisper.** { *; }
```

It's recommended to use `ndkVersion = "24.0.8215888"` (or above) in your root project build configuration for Apple Silicon Macs. Otherwise please follow this trobleshooting [issue](./TROUBLESHOOTING.md#android-got-build-error-unknown-host-cpu-architecture-arm64-on-apple-silicon-macs).

By default, `whisper.rn` will use pre-built libraries for Android: the whisper.cpp core (`librnwhisper*.so`, one per CPU-feature variant, including the Hexagon NPU variant) comes from `android/src/main/jniLibs`, and only the small JNI/JSI wrapper is compiled against your React Native version. If you want to build from source, please set `rnwhisperBuildFromSource` to `true` in `android/gradle.properties` (or pass `-PrnwhisperBuildFromSource=true`). Building from source compiles whisper.cpp once per variant, so it takes a while; `rnwhisperVariants=rnwhisper,rnwhisper_v8fp16_va_2` narrows it down.

##### Hexagon NPU (Experimental)

On Snapdragon devices with a Hexagon Tensor Processor (SM8450 / 8 Gen 1 and newer), `whisper.rn` can run the whisper model on the NPU through ggml's Hexagon backend. It is used automatically when `useGpu` is on (the default) and the device qualifies; check `WhisperContext.gpu` / `reasonNoGPU` after `initWhisper`. The NPU path always uses flash attention. VAD and Parakeet contexts stay on the CPU.

The pre-built libraries include everything the backend needs: the `rnwhisper_v8fp16_va_2_hexagon` variant, and the DSP-side libraries `libggml-htp-*.so` in the package's `bin/arm64-v8a/`, which `whisper.rn`'s Gradle build copies into your app's `src/main/assets/rnwhisper-hexagon/` (the library extracts them at runtime and points the DSP loader at them). Just add the FastRPC loader to your app manifest so the library can open it at runtime:

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
```

When building from source, the Hexagon variant is only compiled if the Hexagon SDK is on the build machine (`scripts/setup-hexagon-sdk.sh` installs it to `~/.hexagon-sdk/6.4.0.2`, or set `HEXAGON_SDK_ROOT`), and the DSP-side libraries have to be built with `npm run build:hexagon-htp` (Docker, using the `ghcr.io/snapdragon-toolchain/arm64-android` image). Without the SDK the from-source build is CPU-only. The example app does this in `example/android/app/build.gradle` (`prepareHTP`) and its manifest.

Benchmark on Samsung S25 Ultra (Snapdragon 8 Elite, from the example app's Benchmark screen, times in ms):

| Model | Th | FA | Enc. | Dec. | Bch5 | PP |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 4 | 1 | 56.04 | 3.40 | 1.01 | 0.30 |
| tiny-q8_0 | 4 | 1 | 56.76 | 3.00 | 0.75 | 0.05 |
| base | 4 | 1 | 83.38 | 3.56 | 1.27 | 0.36 |
| base-q8_0 | 4 | 1 | 84.69 | 2.97 | 0.81 | 0.07 |
| small | 4 | 1 | 190.34 | 8.66 | 3.39 | 0.55 |
| small-q8_0 | 4 | 1 | 190.14 | 9.71 | 2.92 | 0.17 |
| medium | 4 | 1 | 416.22 | 25.91 | 10.09 | 0.89 |
| medium-q8_0 | 4 | 1 | 408.30 | 18.96 | 6.96 | 0.42 |
| large-v3-turbo | 4 | 1 | 705.16 | 8.65 | 2.99 | 0.70 |
| large-v3-turbo-q8_0 | 4 | 1 | 612.35 | 5.62 | 1.78 | 0.11 |

> q5_0 / q5_1 are not supported by ggml-hexagon (they fall back to the CPU), so they are not included.

For Linux arm64 Snapdragon devices (e.g. IQ-9075), [whisper.node](https://github.com/mybigday/whisper.node#snapdragon-hexagon-npu-benchmark) provides the same Hexagon NPU support through its `snapdragon` variant.

#### Expo

You will need to prebuild the project before using it. See [Expo guide](https://docs.expo.dev/workflow/using-libraries/#third-party-libraries) for more details.

## Tips & Tricks

The [Tips & Tricks](docs/TIPS.md) document is a collection of tips and tricks for using `whisper.rn`.

## Usage

```js
import { initWhisper } from 'whisper.rn'

const whisperContext = await initWhisper({
  filePath: 'file://.../ggml-tiny.en.bin',
})

const sampleFilePath = 'file://.../sample.wav'
const options = { language: 'en' }
const { stop, promise } = whisperContext.transcribe(sampleFilePath, options)

const { result } = await promise
// result: (The inference text result from audio file)
```

Whisper always encodes a 30 s window. For short clips, `audioCtx` shrinks the encoder context to speed things up, at some cost in accuracy: pass a fixed size (0 = model default, up to 1500 for stock models), or `'auto'` to fit it to the input length (~2 s of headroom, aligned to 64, at least 256, capped at the model's context). It works for `transcribe()`, `transcribeData()` and `RealtimeTranscriber`'s `transcribeOptions`. It is ignored (full context) while the Core ML encoder is active on iOS, since that model has a fixed 30 s input.

```js
const { promise } = whisperContext.transcribeData(pcmBuffer, { audioCtx: 'auto' })
```

Each segment also carries `noSpeechProb` and `avgLogprob` (the mean log probability of its tokens), signals to drop hallucinated text on silence or noise; pick thresholds for your model and audio. With `language: 'auto'`, `languageProb` is the probability of the detected language. `includeTokens: true` adds the segment's text tokens (a character split across tokens, common in CJK, comes with the token that completes it; the others get empty text). Their times need either `tokenTimestamps: true` (`t0`/`t1`) or a context created with `dtwAheadsPreset` (`tDtw`, aligned from the attention weights and more accurate; not available with flash attention, see `whisperContext.dtw`):

```js
const whisperContext = await initWhisper({ filePath, dtwAheadsPreset: 'base.en' })
const { promise } = whisperContext.transcribe(sampleFilePath, { includeTokens: true })
const { segments } = await promise
segments[0].tokens.forEach(({ text, p, tDtw }) => console.log(text, p, tDtw / 100))
```

## NVIDIA Parakeet TDT

`ParakeetContext` runs NVIDIA's Parakeet TDT 0.6B v3 model through the Parakeet API included in whisper.cpp. The v3 model supports English plus 24 other European languages.

Download a GGUF model from [ggml-org/parakeet-GGUF](https://huggingface.co/ggml-org/parakeet-GGUF) before initializing the context. The example app downloads these models at runtime because they are too large to bundle comfortably:

| Model | Approximate size |
| --- | ---: |
| `ggml-parakeet-tdt-0.6b-v3-q4_0.bin` | 356 MB |
| `ggml-parakeet-tdt-0.6b-v3-q4_k.bin` | 416 MB |
| `ggml-parakeet-tdt-0.6b-v3-q8_0.bin` | 669 MB |
| `ggml-parakeet-tdt-0.6b-v3-f16.bin` | 1.26 GB |

```typescript
import { initParakeet } from 'whisper.rn'

const parakeetContext = await initParakeet({
  filePath: 'file://.../ggml-parakeet-tdt-0.6b-v3-q4_0.bin',
  useGpu: true,
})

const { stop, promise } = parakeetContext.transcribe(
  'file://.../sample.wav',
  { maxThreads: 4 },
)
const { result, segments, isAborted } = await promise

// Cancel an in-flight transcription when needed:
// await stop()

await parakeetContext.release()
```

Parakeet file and base64 inputs must be WAV containing 16-bit PCM audio. `transcribeData()` accepts raw signed 16-bit PCM as a base64 string or `ArrayBuffer`; raw audio must be mono at 16 kHz. Compressed formats such as MP3, AAC, and FLAC are not decoded.

## Voice Activity Detection (VAD)

Voice Activity Detection finds the speech in audio. It runs on [vad.cpp](https://github.com/mybigday/vad.cpp) with its CPU kernels tuned for small models. It supports Silero VAD (whisper.cpp's `ggml-silero-*.bin` files or vad.cpp GGUF files) and FireRedVAD (vad.cpp GGUF).

#### Initialize VAD Context

```typescript
import { initWhisperVad } from 'whisper.rn'

const vadContext = await initWhisperVad({
  filePath: require('./assets/ggml-silero-v6.2.0.bin'), // VAD model file
  nThreads: 1, // Default; the per-hop graphs are too small to gain from more threads
})
```

VAD always runs on the CPU, so `useGpu` is ignored.

#### Detect Speech Segments

##### From Audio Files

```typescript
// Detect speech in audio file (supports same formats as transcribe)
const segments = await vadContext.detectSpeech(require('./assets/audio.wav'), {
  threshold: 0.5, // Speech probability threshold (0.0-1.0)
  minSpeechDurationMs: 250, // Minimum speech duration in ms
  minSilenceDurationMs: 100, // Minimum silence duration in ms
  maxSpeechDurationS: 30, // Maximum speech duration in seconds
  speechPadMs: 30, // Padding around speech segments in ms
})

// Also supports:
// - File paths: vadContext.detectSpeech('path/to/audio.wav', options)
// - HTTP URLs: vadContext.detectSpeech('https://example.com/audio.wav', options)
// - Base64 WAV: vadContext.detectSpeech('data:audio/wav;base64,...', options)
// - Assets: vadContext.detectSpeech(require('./assets/audio.wav'), options)
```

##### From Raw Audio Data

```typescript
// 16 kHz mono 16-bit PCM, as an ArrayBuffer or base64 string
const segments = await vadContext.detectSpeechData(pcmArrayBuffer, {
  threshold: 0.5,
  minSpeechDurationMs: 250,
  minSilenceDurationMs: 100,
  maxSpeechDurationS: 30,
  speechPadMs: 30,
})
```

#### Process Results

```typescript
// t0 / t1 are in 10 ms units
segments.forEach((segment, index) => {
  const start = segment.t0 / 100
  const end = segment.t1 / 100
  console.log(`Segment ${index + 1}: ${start.toFixed(2)}s - ${end.toFixed(2)}s`)
})
```

#### Streaming (stateful) detection

To process live audio, create a stream. It keeps the model state between pushes, so each sample is analyzed once and every probability takes all earlier audio into account. Silero produces one probability per 32 ms hop.

```typescript
const stream = await vadContext.createStream()

// For each incoming chunk of 16 kHz mono PCM16 audio (any length):
const probs = await stream.push(chunk) // speech probabilities that became available
// probs[i] of the whole stream covers samples [i * stream.hopSamples, (i + 1) * stream.hopSamples)

const lastProbs = await stream.flush() // end of audio
await stream.reset() // start over with a new audio stream
await stream.release() // also freed when the context is released
```

`StreamingVadSegmenter` (from `whisper.rn/realtime-transcription`) turns these probabilities into speech segments as they arrive. It uses the same options and segmentation as `detectSpeech*`, except that speech longer than `maxSpeechDurationS` with no pause over 98 ms is split at its least speech-like frame rather than right at the limit (`splitAtDip: false` restores the offline rule). `StreamVad` combines a stream and a segmenter into speech start / continue / end events with the segment's audio; `RealtimeTranscriber` uses it internally (see below).

#### Release VAD Context

```typescript
await vadContext.release()
// Or release all VAD contexts
await releaseAllWhisperVad()
```

## Realtime Transcription

`RealtimeTranscriber` transcribes a live audio stream. A VAD's speech probabilities are segmented into slices, one per stretch of speech. Each slice gets draft transcriptions while it grows, then one final transcription when it ends.

```js
// If your RN packager is not enable package exports support, use whisper.rn/src/realtime-transcription
import { RealtimeTranscriber } from 'whisper.rn/realtime-transcription'
import { AudioPcmStreamAdapter } from 'whisper.rn/realtime-transcription/adapters'
import RNFS from 'react-native-fs' // or any compatible filesystem

const whisperContext = await initWhisper({
  /* ... */
})
// The VAD context itself: the transcriber opens a stateful vad.cpp stream on it
// (one speech probability per 32 ms hop) and segments the probabilities
const vadContext = await initWhisperVad({
  /* ... */
})
const audioStream = new AudioPcmStreamAdapter() // requires @fugood/react-native-audio-pcm-stream

const transcriber = new RealtimeTranscriber(
  { whisperContext, vadContext, audioStream, fs: RNFS },
  {
    vadPreset: 'default', // segmentation: preset + vadOptions overrides
    transcribeOptions: { language: 'en' },
  },
  {
    onTranscribe: (event) => {
      // event.isFinal: final (else a draft a later event replaces)
      // event.startSample / endSample: the slice's audio in the stream
      console.log(event.isFinal ? 'Final:' : 'Draft:', event.data?.result)
    },
    onVad: (event) => console.log('VAD:', event.type, event.confidence),
    onStatusChange: (isActive) =>
      console.log('Status:', isActive ? 'ACTIVE' : 'INACTIVE'),
    onError: (error) => console.error('Error:', error),
  },
)

await transcriber.start()
// stop() transcribes speech still open (endReason 'end_of_stream') before it resolves
await transcriber.stop()
```

- **The VAD:** `vadContext` takes either of two kinds of VAD.
  - **A speech probability source (recommended):** a `WhisperVadContext`, or your own object with `hopSamples`, `sampleRate` and `createStream()`, whose streams `push` audio and return one probability per hop. The transcriber segments the probabilities with `StreamingVadSegmenter`, configured by `vadPreset` / `vadOptions`; pass `probabilitySegmenter` to replace it. This is what enables stateful streaming and splitting long speech back at an earlier pause.
  - **A segmenter (`RealtimeSegmenterLike`):** for example `StreamVad`, the deprecated `RingBufferVad`, or your own VAD that reports speech start / continue / end with the audio. It is used as it is. When its callbacks include sample positions (`VadSpeechChunkInfo` / `VadSpeechEndInfo`), slices are cut exactly at its segment ends; otherwise they follow its events, bounded by `maxSpeechDurationS`.
- **Without a VAD:** audio is cut into slices of at most `vadOptions.maxSpeechDurationS` (default 30 s), at the quietest point near the limit.
- **Drafts:** they run every `realtimeProcessingPauseMs` of audio (default 200), after `initRealtimeAfterMs` (default 200), and can be made cheaper with `draftTranscribeOptions`. For example, `{ audioCtx: 'auto', maxTokens: 64, singleSegment: true, temperatureInc: 0 }`. A final always runs before a waiting draft and aborts a running one. `onBeginTranscribe` gets `isFinal`, so it can skip drafts under load.
- **Slice length:** `audioSliceSec` and `audioMinSec` are gone. `maxSpeechDurationS` bounds every slice, and so every draft's cost. It must be finite: an unbounded value falls back to 30 s.

To use Parakeet, provide `parakeetContext` instead of `whisperContext`:

```js
const parakeetContext = await initParakeet({
  filePath: 'file://.../ggml-parakeet-tdt-0.6b-v3-q4_0.bin',
})

const transcriber = new RealtimeTranscriber(
  { parakeetContext, vadContext, audioStream, fs: RNFS },
  { transcribeOptions: { maxThreads: 4, audioCtx: 0 } },
  { onTranscribe: (event) => console.log(event.data?.result) },
)
```

`initialPrompt` and `promptPreviousSlices` are Whisper-only and are ignored when using `parakeetContext`. A result that is a repetition loop (mostly repeated 3-grams) is not chained into later prompts, since Whisper would repeat it in every following slice. Neither is a slice that follows a split inside speech (`endReason: 'max_duration'` with no pause at the cut): the prompt would end mid-sentence where the audio goes on, and Whisper then drops or repeats words at the cut, with every model size we measured (base to large-v3-turbo). The slice is transcribed with `initialPrompt` only. Realtime Parakeet audio must be mono, 16 kHz, signed 16-bit PCM.

**Long-running sessions:** `maxSlicesInMemory` only bounds raw audio. Per-slice results (transcript + segments) returned by `getTranscriptionResults()` are kept until `stop()`/`reset()` by default, so a transcriber that runs for hours or days grows without bound. Set `maxResultsInMemory` to keep only the newest N results (older ones are dropped oldest-first) and, when `promptPreviousSlices` is enabled, `maxPromptSlices` to cap how many previous results are appended to each Whisper prompt. Consumers that need every result should collect them from `onTranscribe` as they arrive.

```js
const transcriber = new RealtimeTranscriber(
  { whisperContext, vadContext, audioStream },
  {
    maxSlicesInMemory: 5, // raw audio slices
    maxResultsInMemory: 100, // detailed per-slice results
    promptPreviousSlices: true,
    maxPromptSlices: 3, // previous results per prompt
  },
  callbacks,
)
```

**Long speech:** speech longer than the VAD's `maxSpeechDurationS` is split the way `detectSpeech*` splits it: at the longest earlier pause over 98 ms, even when that audio was already transcribed as a draft. The slice is cut there, and the rest starts the next slice. If a pause is in progress at the limit and it is longer than every earlier one, the segment ends where that pause began. Otherwise, with no pause at all, the split falls on the least speech-like 32 ms frame of the last 3 s (in the second half of the segment, at least 0.5 s before the limit), the quietest one when several are about as low, aiming for a gap between words. Final `onTranscribe` events have `isFinal: true` and an `endReason`:
- `'pause'`: the speaker paused.
- `'max_duration'`: the speech went on, so the text probably ends mid-sentence, even if Whisper ended it with a period.
- `'end_of_stream'`: the audio ended.
- `'manual'`: `nextSlice()` was called.

**Quality options (Whisper):**

- **`minAvgLogprob`:** drops result segments whose average token log probability is below it, typically text hallucinated on noise. Pick it for your model and audio.
- **`languageLock`:** with language auto-detection, the language is passed explicitly once it was detected confidently a few times, which also saves the detection pass. Occasional re-checks on long finals let a real language change unlock it (see `LanguageLockOptions`). `transcriber.lockedLanguage` reports it.
- **`splitOverlapMs`:** for example 400. Around a split inside speech, both slices keep that much extra audio, so a word cut by the split is whole on one side. Each word is kept on the side where it starts, by its token times: DTW times when the context was created with `dtwAheadsPreset`, else token timestamps. Words repeated at the boundary are dropped. Final results of those slices are assembled from their tokens. In host measurements (whisper base, 6 s slices without a VAD) it lowered English WER from 0.206 to 0.168 but raised Japanese and Chinese CER, so it suits languages written with spaces.

**Dependencies:**

- `@fugood/react-native-audio-pcm-stream` for `AudioPcmStreamAdapter`
- Compatible filesystem module (e.g., `react-native-fs`). See [filesystem interface](src/utils/WavFileWriter.ts#L9-L16) for TypeScript definition

**Custom Audio Adapters:**
You can create custom audio stream adapters by implementing the [AudioStreamInterface](src/realtime-transcription/types.ts#L21-L30). This allows integration with different audio sources or custom audio processing pipelines.

**Example:** See [complete example](example/src/RealtimeTranscriber.tsx) for full implementation including file simulation and UI.

Please visit the [Documentation](docs/) for more details.

## Usage with assets

You can also use the model file / audio file from assets:

```js
import { initWhisper } from 'whisper.rn'

const whisperContext = await initWhisper({
  filePath: require('../assets/ggml-tiny.en.bin'),
})

const { stop, promise } = whisperContext.transcribe(
  require('../assets/sample.wav'),
  options,
)

// ...
```

This requires editing the `metro.config.js` to support assets:

```js
// ...
const defaultAssetExts = require('metro-config/src/defaults/defaults').assetExts

module.exports = {
  // ...
  resolver: {
    // ...
    assetExts: [
      ...defaultAssetExts,
      'bin', // whisper.rn: ggml model binary
      'mil', // whisper.rn: CoreML model asset
    ],
  },
}
```

Please note that:

- It will significantly increase the size of the app in release mode.
- The RN packager is not allowed file size larger than 2GB, so it not able to use original f16 `large` model (2.9GB), you can use quantized models instead.

## Core ML support

**_Platform: iOS 15.0+, tvOS 15.0+_**

To use Core ML on iOS, you will need to have the Core ML model files.

The `.mlmodelc` model files is load depend on the ggml model file path. For example, if your ggml model path is `ggml-tiny.en.bin`, the Core ML model path will be `ggml-tiny.en-encoder.mlmodelc`. Please note that the ggml model is still needed as decoder or encoder fallback.

The Core ML models are hosted here: https://huggingface.co/ggerganov/whisper.cpp/tree/main

If you want to download model at runtime, during the host file is archive, you will need to unzip the file to get the `.mlmodelc` directory, you can use library like [react-native-zip-archive](https://github.com/mockingbot/react-native-zip-archive), or host those individual files to download yourself.

The `.mlmodelc` is a directory, usually it includes 5 files (3 required):

```json5
[
  'model.mil',
  'coremldata.bin',
  'weights/weight.bin',
  // Not required:
  // 'metadata.json', 'analytics/coremldata.bin',
]
```

Or just use `require` to bundle that in your app, like the example app does, but this would increase the app size significantly.

```js
const whisperContext = await initWhisper({
  filePath: require('../assets/ggml-tiny.en.bin')
  coreMLModelAsset:
    Platform.OS === 'ios'
      ? {
          filename: 'ggml-tiny.en-encoder.mlmodelc',
          assets: [
            require('../assets/ggml-tiny.en-encoder.mlmodelc/weights/weight.bin'),
            require('../assets/ggml-tiny.en-encoder.mlmodelc/model.mil'),
            require('../assets/ggml-tiny.en-encoder.mlmodelc/coremldata.bin'),
          ],
        }
      : undefined,
})
```

In real world, we recommended to split the asset imports into another platform specific file (e.g. `context-opts.ios.js`) to avoid these unused files in the bundle for Android.

## Run with example

The example app provide a simple UI for testing the functions.

Used Whisper model: `tiny.en` in https://huggingface.co/ggerganov/whisper.cpp
Sample file: `jfk.wav` in https://github.com/ggerganov/whisper.cpp/tree/master/samples

Please follow the [Development Workflow section of contributing guide](./CONTRIBUTING.md#development-workflow) to run the example app.

## Mock `whisper.rn`

We have provided a mock version of `whisper.rn` for testing purpose you can use on Jest:

```js
jest.mock('whisper.rn', () => require('whisper.rn/jest-mock'))
```

## Apps using `whisper.rn`

- [BRICKS](https://bricks.tools): Our product for building interactive signage in simple way. We provide LLM functions as Generator LLM/Assistant.
- ... (Any Contribution is welcome)

## Node.js binding

- [whisper.node](https://github.com/mybigday/whisper.node): An another Node.js binding of `whisper.cpp` but made API same as `whisper.rn`. It shares the vendored whisper.cpp with `whisper.rn` and supports Metal, Vulkan, CUDA, the Hexagon NPU (Linux arm64 Snapdragon) and WASM/WebGPU.

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## Troubleshooting

See the [troubleshooting](docs/TROUBLESHOOTING.md) if you encounter any problem while using `whisper.rn`.

## License

MIT

---

Made with [create-react-native-library](https://github.com/callstack/react-native-builder-bob)

---

<p align="center">
  <a href="https://bricks.tools">
    <img width="90px" src="https://avatars.githubusercontent.com/u/17320237?s=200&v=4">
  </a>
  <p align="center">
    Built and maintained by <a href="https://bricks.tools">BRICKS</a>.
  </p>
</p>
