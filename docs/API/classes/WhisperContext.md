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
- [transcribeWithNativeMethod](WhisperContext.md#transcribewithnativemethod)

## Constructors

### constructor

• **new WhisperContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeWhisperContext` |

#### Defined in

[index.ts:203](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L203)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:199](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L199)

___

### id

• **id**: `number`

#### Defined in

[index.ts:197](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L197)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:201](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L201)

## Methods

### bench

▸ **bench**(`maxThreads`): `Promise`\<[`BenchResult`](../README.md#benchresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `maxThreads` | `number` |

#### Returns

`Promise`\<[`BenchResult`](../README.md#benchresult)\>

#### Defined in

[index.ts:465](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L465)

___

### release

▸ **release**(): `Promise`\<`void`\>

#### Returns

`Promise`\<`void`\>

#### Defined in

[index.ts:471](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L471)

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
| `promise` | `Promise`\<[`TranscribeResult`](../README.md#transcriberesult)\> | Transcribe result promise |
| `stop` | () => `Promise`\<`void`\> | Stop the transcribe |

#### Defined in

[index.ts:291](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L291)

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
| `promise` | `Promise`\<[`TranscribeResult`](../README.md#transcriberesult)\> |
| `stop` | () => `Promise`\<`void`\> |

#### Defined in

[index.ts:323](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L323)

___

### transcribeRealtime

▸ **transcribeRealtime**(`options?`): `Promise`\<\{ `stop`: () => `Promise`\<`void`\> ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../README.md#transcriberealtimeevent)) => `void`) => `void`  }\>

Transcribe the microphone audio stream, the microphone user permission is required

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | [`TranscribeRealtimeOptions`](../README.md#transcriberealtimeoptions) |

#### Returns

`Promise`\<\{ `stop`: () => `Promise`\<`void`\> ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../README.md#transcriberealtimeevent)) => `void`) => `void`  }\>

#### Defined in

[index.ts:331](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L331)

___

### transcribeWithNativeMethod

▸ `Private` **transcribeWithNativeMethod**(`method`, `data`, `options?`): `Object`

#### Parameters

| Name | Type |
| :------ | :------ |
| `method` | ``"transcribeFile"`` \| ``"transcribeData"`` |
| `data` | `string` |
| `options` | [`TranscribeFileOptions`](../README.md#transcribefileoptions) |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `promise` | `Promise`\<[`TranscribeResult`](../README.md#transcriberesult)\> |
| `stop` | () => `Promise`\<`void`\> |

#### Defined in

[index.ts:213](https://github.com/Shonn-Li/whisper.rn/blob/c1bd3f8/src/index.ts#L213)
