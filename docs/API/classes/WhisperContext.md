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
- [pauseRealtime](WhisperContext.md#pauserealtime)
- [release](WhisperContext.md#release)
- [resumeRealtime](WhisperContext.md#resumerealtime)
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

[index.ts:213](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L213)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:209](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L209)

___

### id

• **id**: `number`

#### Defined in

[index.ts:207](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L207)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:211](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L211)

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

[index.ts:493](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L493)

___

### pauseRealtime

▸ **pauseRealtime**(): `Promise`\<`void`\>

#### Returns

`Promise`\<`void`\>

#### Defined in

[index.ts:511](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L511)

___

### release

▸ **release**(): `Promise`\<`void`\>

#### Returns

`Promise`\<`void`\>

#### Defined in

[index.ts:507](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L507)

___

### resumeRealtime

▸ **resumeRealtime**(): `Promise`\<`void`\>

#### Returns

`Promise`\<`void`\>

#### Defined in

[index.ts:515](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L515)

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

[index.ts:301](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L301)

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

[index.ts:333](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L333)

___

### transcribeRealtime

▸ **transcribeRealtime**(`options?`): `Promise`\<\{ `onVolumeChange`: (`callback`: (`volume`: `number`) => `void`) => `void` ; `stop`: () => `Promise`\<`void`\> ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../README.md#transcriberealtimeevent)) => `void`) => `void`  }\>

Transcribe the microphone audio stream, the microphone user permission is required

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | [`TranscribeRealtimeOptions`](../README.md#transcriberealtimeoptions) |

#### Returns

`Promise`\<\{ `onVolumeChange`: (`callback`: (`volume`: `number`) => `void`) => `void` ; `stop`: () => `Promise`\<`void`\> ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../README.md#transcriberealtimeevent)) => `void`) => `void`  }\>

#### Defined in

[index.ts:344](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L344)

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

[index.ts:219](https://github.com/Shonn-Li/whisper.rn/blob/78d762f/src/index.ts#L219)
