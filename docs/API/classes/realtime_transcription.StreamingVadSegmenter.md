[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / StreamingVadSegmenter

# Class: StreamingVadSegmenter

[realtime-transcription](../modules/realtime_transcription.md).StreamingVadSegmenter

Incremental version of silero-vad's `get_speech_timestamps_from_probs`, the
segmentation vad.cpp and `WhisperVadContext.detectSpeech*` use offline.

It consumes speech probabilities as a `WhisperVadStream` produces them (one
per hop) and reports segments in absolute samples as soon as they are
certain:
- hysteresis: speech starts at `prob >= threshold` and only a run of
  `prob < negThreshold` lasting `minSilenceDurationMs` ends it;
- `minSpeechDurationMs`: a segment is reported once it is certain to be long
  enough, i.e. exactly the segments the offline pass keeps;
- `speechPadMs`: starts and ends are padded; where the paddings of two
  segments would meet, the pause between them is split in half. An end is
  therefore reported once that is decided: when the next segment is
  confirmed, or `2 * speechPadMs` into the pause;
- `maxSpeechDurationS`: a longer segment is split, like offline, at the
  longest earlier pause of more than 98 ms; the next segment starts where
  that pause ended. Both events then lie in the past, so the consumer must
  keep the open segment's audio (from `retainFrom`) to cut it there.
  Without such a pause see `StreamingVadSegmenterOptions.splitAtDip`.

With `splitAtDip: false` the segments equal the offline result.

## Table of contents

### Constructors

- [constructor](realtime_transcription.StreamingVadSegmenter.md#constructor)

### Accessors

- [inSpeech](realtime_transcription.StreamingVadSegmenter.md#inspeech)
- [maySplitBack](realtime_transcription.StreamingVadSegmenter.md#maysplitback)
- [processedSamples](realtime_transcription.StreamingVadSegmenter.md#processedsamples)
- [retainFrom](realtime_transcription.StreamingVadSegmenter.md#retainfrom)
- [speechCertainUntil](realtime_transcription.StreamingVadSegmenter.md#speechcertainuntil)

### Methods

- [flush](realtime_transcription.StreamingVadSegmenter.md#flush)
- [push](realtime_transcription.StreamingVadSegmenter.md#push)
- [reset](realtime_transcription.StreamingVadSegmenter.md#reset)
- [setFrameEnergy](realtime_transcription.StreamingVadSegmenter.md#setframeenergy)
- [setOptions](realtime_transcription.StreamingVadSegmenter.md#setoptions)

## Constructors

### constructor

• **new StreamingVadSegmenter**(`options`, `hopSamples`, `sampleRate?`, `segmenterOptions?`)

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `options` | [`VadOptions`](../modules/index.md#vadoptions) | `undefined` |
| `hopSamples` | `number` | `undefined` |
| `sampleRate` | `number` | `16000` |
| `segmenterOptions` | [`StreamingVadSegmenterOptions`](../modules/realtime_transcription.md#streamingvadsegmenteroptions) | `{}` |

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:145](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L145)

## Accessors

### inSpeech

• `get` **inSpeech**(): `boolean`

Whether a confirmed speech segment is open (its end not reported yet)

#### Returns

`boolean`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:178](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L178)

___

### maySplitBack

• `get` **maySplitBack**(): `boolean`

Whether segment bounds may still move back into audio already consumed:
an open segment may be split, or the end of one split at a pause in
progress waits for the next start, which can come sooner than
`minSilenceDurationMs` and take part of the padding already consumed.

#### Returns

`boolean`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:235](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L235)

___

### processedSamples

• `get` **processedSamples**(): `number`

Samples covered by the probabilities consumed so far

#### Returns

`number`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:183](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L183)

___

### retainFrom

• `get` **retainFrom**(): `number`

Earliest sample a future or pending segment may still need. While a
segment that may be split is open, that is all of it: a split can move
both its end and the next start back into audio already consumed.

#### Returns

`number`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:224](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L224)

___

### speechCertainUntil

• `get` **speechCertainUntil**(): `number`

End of the audio that surely belongs to the open speech segment (padded),
unless a max-duration split later moves the end back.
Only meaningful while `inSpeech`.

#### Returns

`number`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:192](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L192)

## Methods

### flush

▸ **flush**(`totalSamples`, `probs?`): [`VadSegmenterEvent`](../modules/realtime_transcription.md#vadsegmenterevent)[]

End of audio after `totalSamples` samples: consumes the last
probabilities (the final hop may be partial) and closes open speech.

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `totalSamples` | `number` | `undefined` |
| `probs` | `ArrayLike`<`number`\> | `[]` |

#### Returns

[`VadSegmenterEvent`](../modules/realtime_transcription.md#vadsegmenterevent)[]

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:276](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L276)

___

### push

▸ **push**(`probs`): [`VadSegmenterEvent`](../modules/realtime_transcription.md#vadsegmenterevent)[]

Consume the next probabilities, one per hop.

#### Parameters

| Name | Type |
| :------ | :------ |
| `probs` | `ArrayLike`<`number`\> |

#### Returns

[`VadSegmenterEvent`](../modules/realtime_transcription.md#vadsegmenterevent)[]

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:263](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L263)

___

### reset

▸ **reset**(): `void`

#### Returns

`void`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:247](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L247)

___

### setFrameEnergy

▸ **setFrameEnergy**(`frameEnergy`): `void`

Set or replace the energy function of `StreamingVadSegmenterOptions.frameEnergy`

#### Parameters

| Name | Type |
| :------ | :------ |
| `frameEnergy` | `undefined` \| (`start`: `number`, `end`: `number`) => `number` |

#### Returns

`void`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:243](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L243)

___

### setOptions

▸ **setOptions**(`options`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | [`VadOptions`](../modules/index.md#vadoptions) |

#### Returns

`void`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:159](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/StreamingVadSegmenter.ts#L159)
