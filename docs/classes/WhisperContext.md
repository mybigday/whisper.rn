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

[index.ts:134](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L134)

## Properties

### id

• **id**: `number`

#### Defined in

[index.ts:132](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L132)

## Methods

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:254](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L254)

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

[index.ts:139](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L139)

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

[index.ts:153](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L153)
