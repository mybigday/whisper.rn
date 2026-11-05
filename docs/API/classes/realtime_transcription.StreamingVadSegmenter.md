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
- `maxSpeechDurationS`: a longer segment is split, at the start of the
  current pause if there is one, else right away (the next segment then
  starts at the split). Unlike the offline pass it cannot go back to the
  longest earlier pause, since that audio is out.

Apart from the split rule, the segments equal the offline result.

## Table of contents

### Constructors

- [constructor](realtime_transcription.StreamingVadSegmenter.md#constructor)

### Accessors

- [inSpeech](realtime_transcription.StreamingVadSegmenter.md#inspeech)
- [processedSamples](realtime_transcription.StreamingVadSegmenter.md#processedsamples)
- [retainFrom](realtime_transcription.StreamingVadSegmenter.md#retainfrom)
- [speechCertainUntil](realtime_transcription.StreamingVadSegmenter.md#speechcertainuntil)

### Methods

- [flush](realtime_transcription.StreamingVadSegmenter.md#flush)
- [push](realtime_transcription.StreamingVadSegmenter.md#push)
- [reset](realtime_transcription.StreamingVadSegmenter.md#reset)
- [setOptions](realtime_transcription.StreamingVadSegmenter.md#setoptions)

## Constructors

### constructor

• **new StreamingVadSegmenter**(`options`, `hopSamples`, `sampleRate?`)

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `options` | [`VadOptions`](../modules/index.md#vadoptions) | `undefined` |
| `hopSamples` | `number` | `undefined` |
| `sampleRate` | `number` | `16000` |

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:85](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L85)

## Accessors

### inSpeech

• `get` **inSpeech**(): `boolean`

Whether a confirmed speech segment is open (its end not reported yet)

#### Returns

`boolean`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:110](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L110)

___

### processedSamples

• `get` **processedSamples**(): `number`

Samples covered by the probabilities consumed so far

#### Returns

`number`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:115](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L115)

___

### retainFrom

• `get` **retainFrom**(): `number`

Earliest sample a future or pending segment may still need

#### Returns

`number`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:135](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L135)

___

### speechCertainUntil

• `get` **speechCertainUntil**(): `number`

End of the audio that surely belongs to the open speech segment (padded).
Only meaningful while `inSpeech`.

#### Returns

`number`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:123](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L123)

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

[realtime-transcription/StreamingVadSegmenter.ts:166](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L166)

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

[realtime-transcription/StreamingVadSegmenter.ts:153](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L153)

___

### reset

▸ **reset**(): `void`

#### Returns

`void`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:140](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L140)

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

[realtime-transcription/StreamingVadSegmenter.ts:91](https://github.com/mybigday/whisper.rn/blob/30c029e/src/realtime-transcription/StreamingVadSegmenter.ts#L91)
