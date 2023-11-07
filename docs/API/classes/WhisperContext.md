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

- [release](WhisperContext.md#release)
- [transcribe](WhisperContext.md#transcribe)
- [transcribeRealtime](WhisperContext.md#transcriberealtime)

## Constructors

### constructor

• **new WhisperContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeWhisperContext` |

#### Defined in

[index.ts:190](https://github.com/mybigday/whisper.rn/blob/eed3c2b/src/index.ts#L190)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:186](https://github.com/mybigday/whisper.rn/blob/eed3c2b/src/index.ts#L186)

___

### id

• **id**: `number`

#### Defined in

[index.ts:184](https://github.com/mybigday/whisper.rn/blob/eed3c2b/src/index.ts#L184)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:188](https://github.com/mybigday/whisper.rn/blob/eed3c2b/src/index.ts#L188)

## Methods

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:433](https://github.com/mybigday/whisper.rn/blob/eed3c2b/src/index.ts#L433)

___

### transcribe

▸ **transcribe**(`filePath`, `options?`): `Object`

Transcribe audio file

#### Parameters

| Name | Type |
| :------ | :------ |
| `filePath` | `string` \| `number` |
| `options` | [`TranscribeFileOptions`](../README.md#transcribefileoptions) |

#### Returns

`Object`

| Name | Type | Description |
| :------ | :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../README.md#transcriberesult)\> | Transcribe result promise |
| `stop` | () => `Promise`<`void`\> | Stop the transcribe |

#### Defined in

[index.ts:201](https://github.com/mybigday/whisper.rn/blob/eed3c2b/src/index.ts#L201)

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

[index.ts:297](https://github.com/mybigday/whisper.rn/blob/eed3c2b/src/index.ts#L297)
