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

Each callback also gets the position of its audio (`VadSpeechChunkInfo`,
`VadSpeechEndInfo`), counted in samples of the pushed audio, not wall-clock
time. A segment longer than `maxSpeechDurationS` is split like offline, at
its longest earlier pause (or else at the least speech-like frame of the
last seconds), so the split usually lies in audio already handed out: the
end then reports `endSample` before the audio sent so far (the consumer
drops the rest), and the next segment's audio starts again from its own
padded start, repeating that audio.
Input: 16 kHz mono PCM16.

## Implements

- [`RealtimeSegmenterLike`](../interfaces/realtime_transcription.RealtimeSegmenterLike.md)

## Table of contents

### Constructors

- [constructor](realtime_transcription.StreamVad.md#constructor)

### Accessors

- [options](realtime_transcription.StreamVad.md#options)

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

[realtime-transcription/StreamVad.ts:178](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L178)

## Accessors

### options

• `get` **options**(): [`VadOptions`](../modules/index.md#vadoptions)

Current VAD options (preset merged with overrides)

#### Returns

[`VadOptions`](../modules/index.md#vadoptions)

#### Defined in

[realtime-transcription/StreamVad.ts:199](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L199)

## Methods

### flush

▸ **flush**(): `Promise`<`void`\>

End of audio: emits the remaining probabilities and closes open speech.

#### Returns

`Promise`<`void`\>

#### Implementation of

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[flush](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#flush)

#### Defined in

[realtime-transcription/StreamVad.ts:235](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L235)

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

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[onError](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#onerror)

#### Defined in

[realtime-transcription/StreamVad.ts:219](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L219)

___

### onSpeechContinue

▸ **onSpeechContinue**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `data`: `Uint8Array`, `info`: [`VadSpeechChunkInfo`](../interfaces/realtime_transcription.VadSpeechChunkInfo.md)) => `void` |

#### Returns

`void`

#### Implementation of

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[onSpeechContinue](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#onspeechcontinue)

#### Defined in

[realtime-transcription/StreamVad.ts:209](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L209)

___

### onSpeechEnd

▸ **onSpeechEnd**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `info`: [`VadSpeechEndInfo`](../interfaces/realtime_transcription.VadSpeechEndInfo.md)) => `void` |

#### Returns

`void`

#### Implementation of

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[onSpeechEnd](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#onspeechend)

#### Defined in

[realtime-transcription/StreamVad.ts:215](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L215)

___

### onSpeechStart

▸ **onSpeechStart**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `data`: `Uint8Array`, `info`: [`VadSpeechChunkInfo`](../interfaces/realtime_transcription.VadSpeechChunkInfo.md)) => `void` |

#### Returns

`void`

#### Implementation of

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[onSpeechStart](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#onspeechstart)

#### Defined in

[realtime-transcription/StreamVad.ts:203](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L203)

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

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[processAudio](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#processaudio)

#### Defined in

[realtime-transcription/StreamVad.ts:228](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L228)

___

### release

▸ **release**(): `Promise`<`void`\>

Frees the native stream.

#### Returns

`Promise`<`void`\>

#### Implementation of

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[release](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#release)

#### Defined in

[realtime-transcription/StreamVad.ts:251](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L251)

___

### reset

▸ **reset**(): `Promise`<`void`\>

Waits for queued audio, then starts over (model state included).

#### Returns

`Promise`<`void`\>

#### Implementation of

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[reset](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#reset)

#### Defined in

[realtime-transcription/StreamVad.ts:241](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L241)

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

[RealtimeSegmenterLike](../interfaces/realtime_transcription.RealtimeSegmenterLike.md).[updateOptions](../interfaces/realtime_transcription.RealtimeSegmenterLike.md#updateoptions)

#### Defined in

[realtime-transcription/StreamVad.ts:223](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamVad.ts#L223)
