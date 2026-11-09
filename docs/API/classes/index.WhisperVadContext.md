[whisper.rn](../README.md) / [index](../modules/index.md) / WhisperVadContext

# Class: WhisperVadContext

[index](../modules/index.md).WhisperVadContext

## Table of contents

### Constructors

- [constructor](index.WhisperVadContext.md#constructor)

### Properties

- [arch](index.WhisperVadContext.md#arch)
- [gpu](index.WhisperVadContext.md#gpu)
- [hopSamples](index.WhisperVadContext.md#hopsamples)
- [id](index.WhisperVadContext.md#id)
- [reasonNoGPU](index.WhisperVadContext.md#reasonnogpu)
- [sampleRate](index.WhisperVadContext.md#samplerate)

### Methods

- [createStream](index.WhisperVadContext.md#createstream)
- [detectSpeech](index.WhisperVadContext.md#detectspeech)
- [detectSpeechData](index.WhisperVadContext.md#detectspeechdata)
- [release](index.WhisperVadContext.md#release)

## Constructors

### constructor

• **new WhisperVadContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeWhisperVadContext` |

#### Defined in

[index.ts:700](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L700)

## Properties

### arch

• **arch**: `string` = `''`

Model architecture, e.g. `silero`

#### Defined in

[index.ts:698](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L698)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:687](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L687)

___

### hopSamples

• **hopSamples**: `number` = `512`

Samples between consecutive speech probabilities (512 for Silero)

#### Defined in

[index.ts:695](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L695)

___

### id

• **id**: `number`

#### Defined in

[index.ts:685](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L685)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:689](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L689)

___

### sampleRate

• **sampleRate**: `number` = `16000`

Sample rate the model expects (16000)

#### Defined in

[index.ts:692](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L692)

## Methods

### createStream

▸ **createStream**(): `Promise`<[`WhisperVadStream`](index.WhisperVadStream.md)\>

Create a stateful stream for live audio (see `WhisperVadStream`).
`StreamVad` builds speech segments on top of it for `RealtimeTranscriber`.

#### Returns

`Promise`<[`WhisperVadStream`](index.WhisperVadStream.md)\>

#### Defined in

[index.ts:720](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L720)

___

### detectSpeech

▸ **detectSpeech**(`filePathOrBase64`, `options?`): `Promise`<[`VadSegment`](../modules/index.md#vadsegment)[]\>

Detect speech segments in audio file (path or base64 encoded wav file)
base64: need add `data:audio/wav;base64,` prefix

#### Parameters

| Name | Type |
| :------ | :------ |
| `filePathOrBase64` | `string` \| `number` |
| `options` | [`VadOptions`](../modules/index.md#vadoptions) |

#### Returns

`Promise`<[`VadSegment`](../modules/index.md#vadsegment)[]\>

#### Defined in

[index.ts:730](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L730)

___

### detectSpeechData

▸ **detectSpeechData**(`audioData`, `options?`): `Promise`<[`VadSegment`](../modules/index.md#vadsegment)[]\>

Detect speech segments in raw audio data (base64 encoded float32 PCM data or ArrayBuffer)

#### Parameters

| Name | Type |
| :------ | :------ |
| `audioData` | `string` \| `ArrayBuffer` |
| `options` | [`VadOptions`](../modules/index.md#vadoptions) |

#### Returns

`Promise`<[`VadSegment`](../modules/index.md#vadsegment)[]\>

#### Defined in

[index.ts:755](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L755)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:768](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L768)
