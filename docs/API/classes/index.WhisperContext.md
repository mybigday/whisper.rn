[whisper.rn](../README.md) / [index](../modules/index.md) / WhisperContext

# Class: WhisperContext

[index](../modules/index.md).WhisperContext

## Table of contents

### Constructors

- [constructor](index.WhisperContext.md#constructor)

### Properties

- [gpu](index.WhisperContext.md#gpu)
- [id](index.WhisperContext.md#id)
- [ptr](index.WhisperContext.md#ptr)
- [reasonNoGPU](index.WhisperContext.md#reasonnogpu)

### Methods

- [bench](index.WhisperContext.md#bench)
- [release](index.WhisperContext.md#release)
- [transcribe](index.WhisperContext.md#transcribe)
- [transcribeData](index.WhisperContext.md#transcribedata)
- [transcribeRealtime](index.WhisperContext.md#transcriberealtime)

## Constructors

### constructor

• **new WhisperContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeWhisperContext` |

#### Defined in

[index.ts:247](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L247)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:243](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L243)

___

### id

• **id**: `number`

#### Defined in

[index.ts:241](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L241)

___

### ptr

• **ptr**: `number`

#### Defined in

[index.ts:239](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L239)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:245](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L245)

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

[index.ts:578](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L578)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:592](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L592)

___

### transcribe

▸ **transcribe**(`filePathOrBase64`, `options?`): `Object`

Transcribe audio file (path or base64 encoded wav file)
base64: need add `data:audio/wav;base64,` prefix

#### Parameters

| Name | Type |
| :------ | :------ |
| `filePathOrBase64` | `string` \| `number` |
| `options` | [`TranscribeFileOptions`](../modules/index.md#transcribefileoptions) |

#### Returns

`Object`

| Name | Type | Description |
| :------ | :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> | Transcribe result promise |
| `stop` | () => `Promise`<`void`\> | Stop the transcribe |

#### Defined in

[index.ts:341](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L341)

___

### transcribeData

▸ **transcribeData**(`data`, `options?`): `Object`

Transcribe audio data (base64 encoded float32 PCM data or ArrayBuffer)

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `string` \| `ArrayBuffer` \| `SharedArrayBuffer` |
| `options` | [`TranscribeFileOptions`](../modules/index.md#transcribefileoptions) |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> |
| `stop` | () => `Promise`<`void`\> |

#### Defined in

[index.ts:372](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L372)

___

### transcribeRealtime

▸ **transcribeRealtime**(`options?`): `Promise`<{ `stop`: () => `Promise`<`void`\> ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../modules/index.md#transcriberealtimeevent)) => `void`) => `void`  }\>

Transcribe the microphone audio stream, the microphone user permission is required

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | [`TranscribeRealtimeOptions`](../modules/index.md#transcriberealtimeoptions) |

#### Returns

`Promise`<{ `stop`: () => `Promise`<`void`\> ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../modules/index.md#transcriberealtimeevent)) => `void`) => `void`  }\>

#### Defined in

[index.ts:441](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/index.ts#L441)
