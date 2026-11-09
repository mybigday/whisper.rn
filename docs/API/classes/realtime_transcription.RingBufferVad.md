[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RingBufferVad

# Class: RingBufferVad

[realtime-transcription](../modules/realtime_transcription.md).RingBufferVad

**`Deprecated`**

Use `StreamVad`. RingBufferVad re-runs stateless detection over
the last `preRecordingBufferMs` of audio every `inferenceIntervalMs` and
times speech with the wall clock; StreamVad keeps the model state across
chunks and segments exactly, per hop.

## Implements

- `RealtimeVadContextLike`

## Table of contents

### Constructors

- [constructor](realtime_transcription.RingBufferVad.md#constructor)

### Methods

- [flush](realtime_transcription.RingBufferVad.md#flush)
- [onError](realtime_transcription.RingBufferVad.md#onerror)
- [onSpeechContinue](realtime_transcription.RingBufferVad.md#onspeechcontinue)
- [onSpeechEnd](realtime_transcription.RingBufferVad.md#onspeechend)
- [onSpeechStart](realtime_transcription.RingBufferVad.md#onspeechstart)
- [processAudio](realtime_transcription.RingBufferVad.md#processaudio)
- [reset](realtime_transcription.RingBufferVad.md#reset)
- [updateOptions](realtime_transcription.RingBufferVad.md#updateoptions)

## Constructors

### constructor

• **new RingBufferVad**(`vadContext`, `options?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `vadContext` | `WhisperVadContextLike` |
| `options` | `RingBufferVadOptions` |

#### Defined in

[realtime-transcription/RingBufferVad.ts:57](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L57)

## Methods

### flush

▸ **flush**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Implementation of

RealtimeVadContextLike.flush

#### Defined in

[realtime-transcription/RingBufferVad.ts:138](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L138)

___

### onError

▸ **onError**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`error`: `string`) => `void` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.onError

#### Defined in

[realtime-transcription/RingBufferVad.ts:104](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L104)

___

### onSpeechContinue

▸ **onSpeechContinue**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `data`: `Uint8Array`) => `void` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.onSpeechContinue

#### Defined in

[realtime-transcription/RingBufferVad.ts:96](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L96)

___

### onSpeechEnd

▸ **onSpeechEnd**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`) => `void` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.onSpeechEnd

#### Defined in

[realtime-transcription/RingBufferVad.ts:100](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L100)

___

### onSpeechStart

▸ **onSpeechStart**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `data`: `Uint8Array`) => `void` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.onSpeechStart

#### Defined in

[realtime-transcription/RingBufferVad.ts:92](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L92)

___

### processAudio

▸ **processAudio**(`data`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `Uint8Array` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.processAudio

#### Defined in

[realtime-transcription/RingBufferVad.ts:108](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L108)

___

### reset

▸ **reset**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Implementation of

RealtimeVadContextLike.reset

#### Defined in

[realtime-transcription/RingBufferVad.ts:161](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L161)

___

### updateOptions

▸ **updateOptions**(`options`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Partial`<[`VadOptions`](../modules/index.md#vadoptions)\> |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.updateOptions

#### Defined in

[realtime-transcription/RingBufferVad.ts:286](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/RingBufferVad.ts#L286)
