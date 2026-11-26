[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeVadContextLike

# Interface: RealtimeVadContextLike

[realtime-transcription](../modules/realtime_transcription.md).RealtimeVadContextLike

## Implemented by

- [`RingBufferVad`](../classes/realtime_transcription.RingBufferVad.md)
- [`StreamVad`](../classes/realtime_transcription.StreamVad.md)

## Table of contents

### Properties

- [onError](realtime_transcription.RealtimeVadContextLike.md#onerror)
- [onSpeechContinue](realtime_transcription.RealtimeVadContextLike.md#onspeechcontinue)
- [onSpeechEnd](realtime_transcription.RealtimeVadContextLike.md#onspeechend)
- [onSpeechStart](realtime_transcription.RealtimeVadContextLike.md#onspeechstart)

### Methods

- [flush](realtime_transcription.RealtimeVadContextLike.md#flush)
- [processAudio](realtime_transcription.RealtimeVadContextLike.md#processaudio)
- [release](realtime_transcription.RealtimeVadContextLike.md#release)
- [reset](realtime_transcription.RealtimeVadContextLike.md#reset)
- [updateOptions](realtime_transcription.RealtimeVadContextLike.md#updateoptions)

## Properties

### onError

• **onError**: (`callback`: (`error`: `string`) => `void`) => `void`

#### Type declaration

▸ (`callback`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`error`: `string`) => `void` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:393](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L393)

___

### onSpeechContinue

• **onSpeechContinue**: (`callback`: (`confidence`: `number`, `data`: `Uint8Array`, `info?`: [`VadSpeechChunkInfo`](realtime_transcription.VadSpeechChunkInfo.md)) => `void`) => `void`

#### Type declaration

▸ (`callback`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `data`: `Uint8Array`, `info?`: [`VadSpeechChunkInfo`](realtime_transcription.VadSpeechChunkInfo.md)) => `void` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:387](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L387)

___

### onSpeechEnd

• **onSpeechEnd**: (`callback`: (`confidence`: `number`, `info?`: [`VadSpeechEndInfo`](realtime_transcription.VadSpeechEndInfo.md)) => `void`) => `void`

#### Type declaration

▸ (`callback`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `info?`: [`VadSpeechEndInfo`](realtime_transcription.VadSpeechEndInfo.md)) => `void` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:391](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L391)

___

### onSpeechStart

• **onSpeechStart**: (`callback`: (`confidence`: `number`, `data`: `Uint8Array`, `info?`: [`VadSpeechChunkInfo`](realtime_transcription.VadSpeechChunkInfo.md)) => `void`) => `void`

#### Type declaration

▸ (`callback`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `data`: `Uint8Array`, `info?`: [`VadSpeechChunkInfo`](realtime_transcription.VadSpeechChunkInfo.md)) => `void` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:383](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L383)

## Methods

### flush

▸ **flush**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:397](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L397)

___

### processAudio

▸ **processAudio**(`data`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `Uint8Array` |

#### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:381](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L381)

___

### release

▸ `Optional` **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:401](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L401)

___

### reset

▸ **reset**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:399](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L399)

___

### updateOptions

▸ **updateOptions**(`options`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Partial`<[`VadOptions`](../modules/index.md#vadoptions)\> |

#### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:395](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L395)
