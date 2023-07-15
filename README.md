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
| (tiny.en, Core ML enabled) | (tiny.en, armv8.2-a+fp16) |

## Installation

```sh
npm install whisper.rn
```

Then re-run `npx pod-install` again for iOS.

For Expo, you will need to prebuild the project before using it. See [Expo guide](https://docs.expo.io/guides/using-libraries/#using-a-library-in-a-expo-project) for more details.

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

Please note that it will significantly increase the size of the app in release mode.

## Core ML support

__*Platform: iOS 15.0+, tvOS 15.0+*__

To use Core ML on iOS, you will need to have the Core ML model files.

The `.mlmodelc` model files is load depend on the ggml model file path. For example, if your ggml model path is `ggml-tiny.en.bin`, the Core ML model path will be `ggml-tiny.en-encoder.mlmodelc`. Please note that the ggml model is still needed as decoder or encoder fallback.

Currently there is no official way to get the Core ML models by URL, you will need to convert Core ML models by yourself. Please see [Core ML Support](https://github.com/ggerganov/whisper.cpp#core-ml-support) of whisper.cpp for more details.

During the `.mlmodelc` is a directory, you will need to download 5 files (3 required):

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

Used Whisper model: `tiny.en` in https://huggingface.co/datasets/ggerganov/whisper.cpp  
Sample file: `jfk.wav` in https://github.com/ggerganov/whisper.cpp/tree/master/samples

For test better performance on transcribe, you can run the app in Release mode.
  - iOS: `yarn example ios --configuration Release`
  - Android: `yarn example android --mode release`

Please follow the [Development Workflow section of contributing guide](./CONTRIBUTING.md#development-workflow) to run the example app.

## Mock `whisper.rn`

We have provided a mock version of `whisper.rn` for testing purpose you can use on Jest:

```js
jest.mock('whisper.rn', () => require('whisper.rn/jest/mock'))
```

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## Troubleshooting

See the [troubleshooting](TROUBLESHOOTING.md) if you encounter any problem while using `whisper.rn`.

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
