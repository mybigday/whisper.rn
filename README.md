# whisper.rn

[![Actions Status](https://github.com/mybigday/whisper.rn/workflows/CI/badge.svg)](https://github.com/mybigday/whisper.rn/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/npm/v/whisper.rn.svg)](https://www.npmjs.com/package/whisper.rn/)

React Native binding of [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

[whisper.cpp](https://github.com/ggerganov/whisper.cpp): High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model

## Screenshots

| <img src="https://github.com/mybigday/whisper.rn/assets/3001525/2fea7b2d-c911-44fb-9afc-8efc7b594446" width="300" /> | <img src="https://github.com/mybigday/whisper.rn/assets/3001525/a5005a6c-44f7-4db9-95e8-0fd951a2e147" width="300" /> |
| :------------------------------------------: | :------------------------------------------: |
| iOS: Tested on iPhone 13 Pro Max | Android: Tested on Pixel 6 |
| (tiny.en, Core ML enabled, release mode + archive) | (tiny.en, armv8.2-a+fp16, release mode) |

## Installation

```sh
npm install whisper.rn
```

#### iOS

Please re-run `npx pod-install` again.

#### Android

If you want to use `medium` or `large` model, the [Extended Virtual Addressing](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_extended-virtual-addressing) capability is recommended to enable on iOS project.

For Android, it's recommended to use `ndkVersion = "24.0.8215888"` (or above) in your root project build configuration for Apple Silicon Macs. Otherwise please follow this trobleshooting [issue](./TROUBLESHOOTING.md#android-got-build-error-unknown-host-cpu-architecture-arm64-on-apple-silicon-macs).

Don't forget to add proguard rule if it's enabled in project (android/app/proguard-rules.pro):

```proguard
# whisper.rn
-keep class com.rnwhisper.** { *; }
```

#### Expo

You will need to prebuild the project before using it. See [Expo guide](https://docs.expo.io/guides/using-libraries/#using-a-library-in-a-expo-project) for more details.

## Add Microphone Permissions (Optional)

If you want to use realtime transcribe, you need to add the microphone permission to your app.

### iOS
Add these lines to ```ios/[YOU_APP_NAME]/info.plist```
```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app requires microphone access in order to transcribe speech</string>
```

For tvOS, please note that the microphone is not supported.

### Android
Add the following line to ```android/app/src/main/AndroidManifest.xml```
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```
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

Use realtime transcribe:

```js
const { stop, subscribe } = await whisperContext.transcribeRealtime(options)

subscribe(evt => {
  const { isCapturing, data, processTime, recordingTime } = evt
  console.log(
    `Realtime transcribing: ${isCapturing ? 'ON' : 'OFF'}\n` +
      // The inference text result from audio record:
      `Result: ${data.result}\n\n` + 
      `Process time: ${processTime}ms\n` +
      `Recording time: ${recordingTime}ms`,
  )
  if (!isCapturing) console.log('Finished realtime transcribing')
})
```

In iOS, You may need to change the Audio Session so that it can be used with other audio playback, or to optimize the quality of the recording. So we have provided AudioSession utilities for you:

Option 1 - Use options in transcribeRealtime:
```js
import { AudioSessionIos } from 'whisper.rn'

const { stop, subscribe } = await whisperContext.transcribeRealtime({
  audioSessionOnStartIos: {
    category: AudioSessionIos.Category.PlayAndRecord,
    options: [AudioSessionIos.CategoryOption.MixWithOthers],
    mode: AudioSessionIos.Mode.Default,
  },
  audioSessionOnStopIos: 'restore', // Or an AudioSessionSettingIos
})
```

Option 2 - Manage the Audio Session in anywhere:
```js
import { AudioSessionIos } from 'whisper.rn'

await AudioSessionIos.setCategory(
  AudioSessionIos.Category.PlayAndRecord, [AudioSessionIos.CategoryOption.MixWithOthers],
)
await AudioSessionIos.setMode(AudioSessionIos.Mode.Default)
await AudioSessionIos.setActive(true)
// Then you can start do recording
```

In Android, you may need to request the microphone permission by [`PermissionAndroid`](https://reactnative.dev/docs/permissionsandroid).

Please visit the [Documentation](docs/) for more details.

## Usage with assets

You can also use the model file / audio file from assets:

```js
import { initWhisper } from 'whisper.rn'

const whisperContext = await initWhisper({
  filePath: require('../assets/ggml-tiny.en.bin'),
})

const { stop, promise } =
  whisperContext.transcribe(require('../assets/sample.wav'), options)

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
    ]
  },
}
```

Please note that:
- It will significantly increase the size of the app in release mode.
- The RN packager is not allowed file size larger than 2GB, so it not able to use original f16 `large` model (2.9GB), you can use quantized models instead.

## Core ML support

__*Platform: iOS 15.0+, tvOS 15.0+*__

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
jest.mock('whisper.rn', () => require('whisper.rn/jest/mock'))
```

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
