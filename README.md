# whisper.rn

[![Actions Status](https://github.com/mybigday/whisper.rn/workflows/CI/badge.svg)](https://github.com/mybigday/whisper.rn/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/npm/v/whisper.rn.svg)](https://www.npmjs.com/package/whisper.rn/)

React Native binding of [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

[whisper.cpp](https://github.com/ggerganov/whisper.cpp): High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model

<img src="https://user-images.githubusercontent.com/3001525/225511664-8b2ba3ec-864d-4f55-bcb0-447aef168a32.jpeg" width="500" />

> Run example with release mode on iPhone 13 Pro Max

## Installation

```sh
npm install whisper.rn
```

Then re-run `npx pod-install` again for iOS.

## Usage

```js
import { initWhisper } from 'whisper.rn'

const filePath = 'file://.../ggml.base.en.bin'
const sampleFilePath = 'file://.../sample.wav'

const whisperContext = await initWhisper({ filePath })

const { result } = await whisperContext.transcribe(sampleFilePath, {
  language: 'en',
  // More options
})
// result: (The inference text result from audio file)
```

## Run with example

The example app is using [react-native-fs](https://github.com/itinance/react-native-fs) to download the model file and audio file.

Model: `base.en` in https://huggingface.co/datasets/ggerganov/whisper.cpp
Sample file: `jfk.wav` in https://github.com/ggerganov/whisper.cpp/tree/master/samples

For test better performance on transcribe, you can run the app in Release mode.
  - iOS: `yarn example ios --configuration Release`
  - Android: `yarn example android --mode release`

Please follow [CONTIBUTING.md](./CONTRIBUTING.md#development-workflow) to run the example app.

## Mock `whisper.rn`

We have provided a mock version of `whisper.rn` for testing purpose you can use on Jest:

```js
jest.mock('whisper.rn', () => require('whisper.rn/jest/mock'))
```

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

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
