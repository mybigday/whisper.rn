[whisper.rn](../README.md) / WhisperContext

# Class: WhisperContext

## Table of contents

### Constructors

- [constructor](WhisperContext.md#constructor)

### Properties

- [gpu](WhisperContext.md#gpu)
- [id](WhisperContext.md#id)
- [reasonNoGPU](WhisperContext.md#reasonnogpu)

### Methods

- [bench](WhisperContext.md#bench)
- [release](WhisperContext.md#release)
- [transcribe](WhisperContext.md#transcribe)
- [transcribeData](WhisperContext.md#transcribedata)
- [transcribeRealtime](WhisperContext.md#transcriberealtime)

## Constructors

### constructor

• **new WhisperContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeWhisperContext` |

#### Defined in

[index.ts:208](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L208)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:204](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L204)

___

### id

• **id**: `number`

#### Defined in

[index.ts:202](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L202)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:206](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L206)

## Methods

### bench

▸ **bench**(`maxThreads`): `Promise`<[`BenchResult`](../README.md#benchresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `maxThreads` | `number` |

#### Returns

`Promise`<[`BenchResult`](../README.md#benchresult)\>

#### Defined in

[index.ts:469](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L469)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:475](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L475)

___

### transcribe

▸ **transcribe**(`filePathOrBase64`, `options?`): `Object`

Transcribe audio file (path or base64 encoded wav file)
base64: need add `data:audio/wav;base64,` prefix

#### Parameters

| Name | Type |
| :------ | :------ |
| `filePathOrBase64` | `string` \| `number` |
| `options` | [`TranscribeFileOptions`](../README.md#transcribefileoptions) |

#### Returns

`Object`

| Name | Type | Description |
| :------ | :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../README.md#transcriberesult)\> | Transcribe result promise |
| `stop` | () => `Promise`<`void`\> | Stop the transcribe |

#### Defined in

[index.ts:296](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L296)

___

### transcribeData

▸ **transcribeData**(`data`, `options?`): `Object`

Transcribe audio data (base64 encoded float32 PCM data)

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `string` |
| `options` | [`TranscribeFileOptions`](../README.md#transcribefileoptions) |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../README.md#transcriberesult)\> |
| `stop` | () => `Promise`<`void`\> |

#### Defined in

[index.ts:327](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L327)

___

### transcribeRealtime

▸ **transcribeRealtime**(`options?`): `Promise`<{ `stop`: () => `Promise`<`void`\> ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../README.md#transcriberealtimeevent)) => `void`) => `void`  }\>

Transcribe the microphone audio stream, the microphone user permission is required

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | [`TranscribeRealtimeOptions`](../README.md#transcriberealtimeoptions) |

#### Returns

`Promise`<{ `stop`: () => `Promise`<`void`\> ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../README.md#transcriberealtimeevent)) => `void`) => `void`  }\>

#### Defined in

[index.ts:335](https://github.com/mybigday/whisper.rn/blob/a11299e/src/index.ts#L335)
