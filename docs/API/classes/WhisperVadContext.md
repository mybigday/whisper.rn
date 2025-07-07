[whisper.rn](../README.md) / WhisperVadContext

# Class: WhisperVadContext

## Table of contents

### Constructors

- [constructor](WhisperVadContext.md#constructor)

### Properties

- [gpu](WhisperVadContext.md#gpu)
- [id](WhisperVadContext.md#id)
- [reasonNoGPU](WhisperVadContext.md#reasonnogpu)

### Methods

- [detectSpeech](WhisperVadContext.md#detectspeech)
- [detectSpeechData](WhisperVadContext.md#detectspeechdata)
- [release](WhisperVadContext.md#release)

## Constructors

### constructor

• **new WhisperVadContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeWhisperVadContext` |

#### Defined in

[index.ts:729](https://github.com/mybigday/whisper.rn/blob/a6284b1/src/index.ts#L729)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:725](https://github.com/mybigday/whisper.rn/blob/a6284b1/src/index.ts#L725)

___

### id

• **id**: `number`

#### Defined in

[index.ts:723](https://github.com/mybigday/whisper.rn/blob/a6284b1/src/index.ts#L723)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:727](https://github.com/mybigday/whisper.rn/blob/a6284b1/src/index.ts#L727)

## Methods

### detectSpeech

▸ **detectSpeech**(`filePathOrBase64`, `options?`): `Promise`<[`VadSegment`](../README.md#vadsegment)[]\>

Detect speech segments in audio file (path or base64 encoded wav file)
base64: need add `data:audio/wav;base64,` prefix

#### Parameters

| Name | Type |
| :------ | :------ |
| `filePathOrBase64` | `string` \| `number` |
| `options` | [`VadOptions`](../README.md#vadoptions) |

#### Returns

`Promise`<[`VadSegment`](../README.md#vadsegment)[]\>

#### Defined in

[index.ts:739](https://github.com/mybigday/whisper.rn/blob/a6284b1/src/index.ts#L739)

___

### detectSpeechData

▸ **detectSpeechData**(`audioData`, `options?`): `Promise`<[`VadSegment`](../README.md#vadsegment)[]\>

Detect speech segments in raw audio data (base64 encoded float32 PCM data or ArrayBuffer)

#### Parameters

| Name | Type |
| :------ | :------ |
| `audioData` | `string` \| `ArrayBuffer` \| `SharedArrayBuffer` |
| `options` | [`VadOptions`](../README.md#vadoptions) |

#### Returns

`Promise`<[`VadSegment`](../README.md#vadsegment)[]\>

#### Defined in

[index.ts:773](https://github.com/mybigday/whisper.rn/blob/a6284b1/src/index.ts#L773)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:795](https://github.com/mybigday/whisper.rn/blob/a6284b1/src/index.ts#L795)
