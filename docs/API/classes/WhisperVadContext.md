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

[index.ts:611](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L611)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:607](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L607)

___

### id

• **id**: `number`

#### Defined in

[index.ts:605](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L605)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:609](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L609)

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

[index.ts:625](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L625)

___

### detectSpeechData

▸ **detectSpeechData**(`audioData`, `options?`): `Promise`<[`VadSegment`](../README.md#vadsegment)[]\>

Detect speech segments in raw audio data (base64 encoded float32 PCM data)

#### Parameters

| Name | Type |
| :------ | :------ |
| `audioData` | `string` |
| `options` | [`VadOptions`](../README.md#vadoptions) |

#### Returns

`Promise`<[`VadSegment`](../README.md#vadsegment)[]\>

#### Defined in

[index.ts:659](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L659)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:666](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L666)
