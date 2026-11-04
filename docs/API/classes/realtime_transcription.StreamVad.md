[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / StreamVad

# Class: StreamVad

[realtime-transcription](../modules/realtime_transcription.md).StreamVad

Stateful VAD for `RealtimeTranscriber`, on a `WhisperVadStream`.

Every incoming chunk is pushed to the stream once; the model state carries
over, so there is one exact speech probability per hop (32 ms for Silero)
instead of re-running a window of recent audio. `StreamingVadSegmenter`
turns the probabilities into segments with `VadOptions`, and the speech
events carry the segment's audio:
- `onSpeechStart`: confirmed speech, with the audio from its padded start;
- `onSpeechContinue`: further audio of the segment, each sample once;
- `onSpeechEnd`: after the audio up to the padded end.

Positions are counted in samples of the pushed audio, not wall-clock time.
Input: 16 kHz mono PCM16.

## Implements

- `RealtimeVadContextLike`

## Table of contents

### Constructors

- [constructor](realtime_transcription.StreamVad.md#constructor)

### Methods

- [flush](realtime_transcription.StreamVad.md#flush)
- [onError](realtime_transcription.StreamVad.md#onerror)
- [onSpeechContinue](realtime_transcription.StreamVad.md#onspeechcontinue)
- [onSpeechEnd](realtime_transcription.StreamVad.md#onspeechend)
- [onSpeechStart](realtime_transcription.StreamVad.md#onspeechstart)
- [processAudio](realtime_transcription.StreamVad.md#processaudio)
- [release](realtime_transcription.StreamVad.md#release)
- [reset](realtime_transcription.StreamVad.md#reset)
- [updateOptions](realtime_transcription.StreamVad.md#updateoptions)

## Constructors

### constructor

• **new StreamVad**(`vadContext`, `options?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `vadContext` | [`StreamVadContextLike`](../modules/realtime_transcription.md#streamvadcontextlike) |
| `options` | [`StreamVadOptions`](../modules/realtime_transcription.md#streamvadoptions) |

#### Defined in

[realtime-transcription/StreamVad.ts:140](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L140)

## Methods

### flush

▸ **flush**(): `Promise`<`void`\>

End of audio: emits the remaining probabilities and closes open speech.

#### Returns

`Promise`<`void`\>

#### Implementation of

RealtimeVadContextLike.flush

#### Defined in

[realtime-transcription/StreamVad.ts:180](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L180)

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

[realtime-transcription/StreamVad.ts:164](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L164)

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

[realtime-transcription/StreamVad.ts:156](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L156)

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

[realtime-transcription/StreamVad.ts:160](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L160)

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

[realtime-transcription/StreamVad.ts:152](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L152)

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

[realtime-transcription/StreamVad.ts:173](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L173)

___

### release

▸ **release**(): `Promise`<`void`\>

Frees the native stream.

#### Returns

`Promise`<`void`\>

#### Implementation of

RealtimeVadContextLike.release

#### Defined in

[realtime-transcription/StreamVad.ts:196](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L196)

___

### reset

▸ **reset**(): `Promise`<`void`\>

Waits for queued audio, then starts over (model state included).

#### Returns

`Promise`<`void`\>

#### Implementation of

RealtimeVadContextLike.reset

#### Defined in

[realtime-transcription/StreamVad.ts:186](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L186)

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

[realtime-transcription/StreamVad.ts:168](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L168)
