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

[index.ts:104](https://github.com/mybigday/whisper.rn/blob/22cc9a8/src/index.ts#L104)

## Properties

### id

• **id**: `number`

#### Defined in

[index.ts:102](https://github.com/mybigday/whisper.rn/blob/22cc9a8/src/index.ts#L102)

## Methods

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:271](https://github.com/mybigday/whisper.rn/blob/22cc9a8/src/index.ts#L271)

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
| `stop` | () => `void` | Stop the transcribe |

#### Defined in

[index.ts:109](https://github.com/mybigday/whisper.rn/blob/22cc9a8/src/index.ts#L109)

___

### transcribeRealtime

▸ **transcribeRealtime**(`options?`): `Promise`<{ `stop`: () => `void` ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../README.md#transcriberealtimeevent)) => `void`) => `void`  }\>

Transcribe the microphone audio stream, the microphone user permission is required

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | [`TranscribeRealtimeOptions`](../README.md#transcriberealtimeoptions) |

#### Returns

`Promise`<{ `stop`: () => `void` ; `subscribe`: (`callback`: (`event`: [`TranscribeRealtimeEvent`](../README.md#transcriberealtimeevent)) => `void`) => `void`  }\>

#### Defined in

[index.ts:165](https://github.com/mybigday/whisper.rn/blob/22cc9a8/src/index.ts#L165)
