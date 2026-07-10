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

[index.ts:604](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L604)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:600](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L600)

___

### id

• **id**: `number`

#### Defined in

[index.ts:598](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L598)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:602](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L602)

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

[index.ts:614](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L614)

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

[index.ts:639](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L639)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:652](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L652)
