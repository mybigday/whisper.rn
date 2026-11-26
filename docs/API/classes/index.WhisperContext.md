[whisper.rn](../README.md) / [index](../modules/index.md) / WhisperContext

# Class: WhisperContext

[index](../modules/index.md).WhisperContext

## Table of contents

### Constructors

- [constructor](index.WhisperContext.md#constructor)

### Properties

- [dtw](index.WhisperContext.md#dtw)
- [gpu](index.WhisperContext.md#gpu)
- [id](index.WhisperContext.md#id)
- [ptr](index.WhisperContext.md#ptr)
- [reasonNoGPU](index.WhisperContext.md#reasonnogpu)

### Methods

- [bench](index.WhisperContext.md#bench)
- [release](index.WhisperContext.md#release)
- [transcribe](index.WhisperContext.md#transcribe)
- [transcribeData](index.WhisperContext.md#transcribedata)

## Constructors

### constructor

• **new WhisperContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeWhisperContext` |

#### Defined in

[index.ts:258](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L258)

## Properties

### dtw

• **dtw**: `boolean` = `false`

Whether DTW token timestamps are computed (`dtwAheadsPreset`, off with flash attention)

#### Defined in

[index.ts:256](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L256)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:251](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L251)

___

### id

• **id**: `number`

#### Defined in

[index.ts:249](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L249)

___

### ptr

• **ptr**: `number`

#### Defined in

[index.ts:247](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L247)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:253](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L253)

## Methods

### bench

▸ **bench**(`maxThreads`): `Promise`<[`BenchResult`](../modules/index.md#benchresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `maxThreads` | `number` |

#### Returns

`Promise`<[`BenchResult`](../modules/index.md#benchresult)\>

#### Defined in

[index.ts:382](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L382)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:388](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L388)

___

### transcribe

▸ **transcribe**(`filePathOrBase64`, `options?`): `Object`

Transcribe audio file (path or base64 encoded wav file)
base64: need add `data:audio/wav;base64,` prefix

#### Parameters

| Name | Type |
| :------ | :------ |
| `filePathOrBase64` | `string` \| `number` |
| `options` | [`TranscribeFileOptions`](../interfaces/index.TranscribeFileOptions.md) |

#### Returns

`Object`

| Name | Type | Description |
| :------ | :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> | Transcribe result promise |
| `stop` | () => `Promise`<`void`\> | Stop the transcribe |

#### Defined in

[index.ts:290](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L290)

___

### transcribeData

▸ **transcribeData**(`data`, `options?`): `Object`

Transcribe audio data (base64 encoded float32 PCM data or ArrayBuffer)

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `string` \| `ArrayBuffer` |
| `options` | [`TranscribeFileOptions`](../interfaces/index.TranscribeFileOptions.md) |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> |
| `stop` | () => `Promise`<`void`\> |

#### Defined in

[index.ts:343](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L343)
