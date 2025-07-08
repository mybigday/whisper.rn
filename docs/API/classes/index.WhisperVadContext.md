[whisper.rn](../README.md) / [index](../modules/index.md) / WhisperVadContext

# Class: WhisperVadContext

[index](../modules/index.md).WhisperVadContext

## Table of contents

### Constructors

- [constructor](index.WhisperVadContext.md#constructor)

### Properties

- [gpu](index.WhisperVadContext.md#gpu)
- [id](index.WhisperVadContext.md#id)
- [reasonNoGPU](index.WhisperVadContext.md#reasonnogpu)

### Methods

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

[index.ts:729](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/index.ts#L729)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:725](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/index.ts#L725)

___

### id

• **id**: `number`

#### Defined in

[index.ts:723](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/index.ts#L723)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:727](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/index.ts#L727)

## Methods

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

[index.ts:739](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/index.ts#L739)

___

### detectSpeechData

▸ **detectSpeechData**(`audioData`, `options?`): `Promise`<[`VadSegment`](../modules/index.md#vadsegment)[]\>

Detect speech segments in raw audio data (base64 encoded float32 PCM data or ArrayBuffer)

#### Parameters

| Name | Type |
| :------ | :------ |
| `audioData` | `string` \| `ArrayBuffer` \| `SharedArrayBuffer` |
| `options` | [`VadOptions`](../modules/index.md#vadoptions) |

#### Returns

`Promise`<[`VadSegment`](../modules/index.md#vadsegment)[]\>

#### Defined in

[index.ts:773](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/index.ts#L773)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:795](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/index.ts#L795)
