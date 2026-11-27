[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeSegmenterLike

# Interface: RealtimeSegmenterLike

[realtime-transcription](../modules/realtime_transcription.md).RealtimeSegmenterLike

A segmenter: turns pushed audio into speech segments, reported as start /
continue / end events with the segment's audio. `StreamVad` and
`RingBufferVad` implement it; other VADs can too. The optional `info`
arguments (sample positions, end reason) let `RealtimeTranscriber` cut slices
exactly and report why they ended.

## Implemented by

- [`RingBufferVad`](../classes/realtime_transcription.RingBufferVad.md)
- [`StreamVad`](../classes/realtime_transcription.StreamVad.md)

## Table of contents

### Properties

- [onError](realtime_transcription.RealtimeSegmenterLike.md#onerror)
- [onSpeechContinue](realtime_transcription.RealtimeSegmenterLike.md#onspeechcontinue)
- [onSpeechEnd](realtime_transcription.RealtimeSegmenterLike.md#onspeechend)
- [onSpeechStart](realtime_transcription.RealtimeSegmenterLike.md#onspeechstart)

### Methods

- [flush](realtime_transcription.RealtimeSegmenterLike.md#flush)
- [processAudio](realtime_transcription.RealtimeSegmenterLike.md#processaudio)
- [release](realtime_transcription.RealtimeSegmenterLike.md#release)
- [reset](realtime_transcription.RealtimeSegmenterLike.md#reset)
- [updateOptions](realtime_transcription.RealtimeSegmenterLike.md#updateoptions)

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

[realtime-transcription/types.ts:403](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L403)

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

[realtime-transcription/types.ts:397](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L397)

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

[realtime-transcription/types.ts:401](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L401)

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

[realtime-transcription/types.ts:393](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L393)

## Methods

### flush

▸ **flush**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:407](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L407)

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

[realtime-transcription/types.ts:391](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L391)

___

### release

▸ `Optional` **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:411](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L411)

___

### reset

▸ **reset**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:409](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L409)

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

[realtime-transcription/types.ts:405](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L405)
