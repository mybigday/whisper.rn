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

[index.ts:737](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L737)

## Properties

### arch

• **arch**: `string` = `''`

Model architecture, e.g. `silero`

#### Defined in

[index.ts:735](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L735)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:724](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L724)

___

### hopSamples

• **hopSamples**: `number` = `512`

Samples between consecutive speech probabilities (512 for Silero)

#### Defined in

[index.ts:732](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L732)

___

### id

• **id**: `number`

#### Defined in

[index.ts:722](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L722)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:726](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L726)

___

### sampleRate

• **sampleRate**: `number` = `16000`

Sample rate the model expects (16000)

#### Defined in

[index.ts:729](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L729)

## Methods

### createStream

▸ **createStream**(): `Promise`<[`WhisperVadStream`](index.WhisperVadStream.md)\>

Create a stateful stream for live audio (see `WhisperVadStream`).
`StreamVad` builds speech segments on top of it for `RealtimeTranscriber`.

#### Returns

`Promise`<[`WhisperVadStream`](index.WhisperVadStream.md)\>

#### Defined in

[index.ts:757](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L757)

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

[index.ts:767](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L767)

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

[index.ts:792](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L792)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:805](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L805)
