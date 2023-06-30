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

[index.ts:82](https://github.com/mybigday/whisper.rn/blob/9b07569/src/index.ts#L82)

## Properties

### id

• **id**: `number`

#### Defined in

[index.ts:80](https://github.com/mybigday/whisper.rn/blob/9b07569/src/index.ts#L80)

## Methods

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:202](https://github.com/mybigday/whisper.rn/blob/9b07569/src/index.ts#L202)

___

### transcribe

▸ **transcribe**(`path`, `options?`): `Object`

Transcribe audio file

#### Parameters

| Name | Type |
| :------ | :------ |
| `path` | `string` |
| `options` | [`TranscribeOptions`](../README.md#transcribeoptions) |

#### Returns

`Object`

| Name | Type | Description |
| :------ | :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../README.md#transcriberesult)\> | Transcribe result promise |
| `stop` | () => `void` | Stop the transcribe |

#### Defined in

[index.ts:87](https://github.com/mybigday/whisper.rn/blob/9b07569/src/index.ts#L87)

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

[index.ts:101](https://github.com/mybigday/whisper.rn/blob/9b07569/src/index.ts#L101)
