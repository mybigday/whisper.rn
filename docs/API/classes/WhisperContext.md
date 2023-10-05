[whisper.rn](../README.md) / WhisperContext

# Class: WhisperContext

## Table of contents

### Constructors

- [constructor](WhisperContext.md#constructor)

### Properties

- [id](WhisperContext.md#id)

### Methods

- [release](WhisperContext.md#release)
- [transcribe](WhisperContext.md#transcribe)
- [transcribeRealtime](WhisperContext.md#transcriberealtime)

## Constructors

### constructor

• **new WhisperContext**(`id`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `id` | `number` |

#### Defined in

[index.ts:186](https://github.com/mybigday/whisper.rn/blob/ca8e344/src/index.ts#L186)

## Properties

### id

• **id**: `number`

#### Defined in

[index.ts:184](https://github.com/mybigday/whisper.rn/blob/ca8e344/src/index.ts#L184)

## Methods

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:423](https://github.com/mybigday/whisper.rn/blob/ca8e344/src/index.ts#L423)

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

[index.ts:191](https://github.com/mybigday/whisper.rn/blob/ca8e344/src/index.ts#L191)

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

[index.ts:287](https://github.com/mybigday/whisper.rn/blob/ca8e344/src/index.ts#L287)
