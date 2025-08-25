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

[index.ts:268](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L268)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:264](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L264)

___

### id

• **id**: `number`

#### Defined in

[index.ts:262](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L262)

___

### ptr

• **ptr**: `number`

#### Defined in

[index.ts:260](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L260)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:266](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L266)

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

[index.ts:603](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L603)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:617](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L617)

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

[index.ts:362](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L362)

___

### transcribeData

▸ **transcribeData**(`data`, `options?`): `Object`

Transcribe audio data (base64 encoded float32 PCM data or ArrayBuffer)

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `string` \| `ArrayBuffer` |
| `options` | [`TranscribeFileOptions`](../modules/index.md#transcribefileoptions) |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> |
| `stop` | () => `Promise`<`void`\> |

#### Defined in

[index.ts:393](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L393)

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

[index.ts:462](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/index.ts#L462)
