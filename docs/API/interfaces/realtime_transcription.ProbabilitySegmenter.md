[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / ProbabilitySegmenter

# Interface: ProbabilitySegmenter

[realtime-transcription](../modules/realtime_transcription.md).ProbabilitySegmenter

Turns per-hop speech probabilities into segment events. `StreamVad` (and so
`RealtimeTranscriber`) runs one; `StreamingVadSegmenter` is the default.

## Implemented by

- [`StreamingVadSegmenter`](../classes/realtime_transcription.StreamingVadSegmenter.md)

## Table of contents

### Properties

- [maySplitBack](realtime_transcription.ProbabilitySegmenter.md#maysplitback)
- [retainFrom](realtime_transcription.ProbabilitySegmenter.md#retainfrom)
- [speechCertainUntil](realtime_transcription.ProbabilitySegmenter.md#speechcertainuntil)

### Methods

- [flush](realtime_transcription.ProbabilitySegmenter.md#flush)
- [push](realtime_transcription.ProbabilitySegmenter.md#push)
- [reset](realtime_transcription.ProbabilitySegmenter.md#reset)
- [setFrameEnergy](realtime_transcription.ProbabilitySegmenter.md#setframeenergy)
- [setOptions](realtime_transcription.ProbabilitySegmenter.md#setoptions)

## Properties

### maySplitBack

• `Readonly` **maySplitBack**: `boolean`

Whether an end or next start may still move back into consumed audio

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:56](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamingVadSegmenter.ts#L56)

___

### retainFrom

• `Readonly` **retainFrom**: `number`

Earliest sample a future or pending segment may still need

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:54](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamingVadSegmenter.ts#L54)

___

### speechCertainUntil

• `Readonly` **speechCertainUntil**: `number`

Audio up to here surely belongs to the open segment (emitted as it comes)

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:52](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamingVadSegmenter.ts#L52)

## Methods

### flush

▸ **flush**(`totalSamples`, `probs?`): [`VadSegmenterEvent`](../modules/realtime_transcription.md#vadsegmenterevent)[]

End of audio after `totalSamples` samples, with the last probabilities

#### Parameters

| Name | Type |
| :------ | :------ |
| `totalSamples` | `number` |
| `probs?` | `ArrayLike`<`number`\> |

#### Returns

[`VadSegmenterEvent`](../modules/realtime_transcription.md#vadsegmenterevent)[]

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:48](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamingVadSegmenter.ts#L48)

___

### push

▸ **push**(`probs`): [`VadSegmenterEvent`](../modules/realtime_transcription.md#vadsegmenterevent)[]

#### Parameters

| Name | Type |
| :------ | :------ |
| `probs` | `ArrayLike`<`number`\> |

#### Returns

[`VadSegmenterEvent`](../modules/realtime_transcription.md#vadsegmenterevent)[]

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:46](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamingVadSegmenter.ts#L46)

___

### reset

▸ **reset**(): `void`

#### Returns

`void`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:49](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamingVadSegmenter.ts#L49)

___

### setFrameEnergy

▸ `Optional` **setFrameEnergy**(`frameEnergy`): `void`

Optional: energy of audio samples, for picking split points

#### Parameters

| Name | Type |
| :------ | :------ |
| `frameEnergy` | `undefined` \| (`start`: `number`, `end`: `number`) => `number` |

#### Returns

`void`

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:58](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamingVadSegmenter.ts#L58)

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

[realtime-transcription/StreamingVadSegmenter.ts:50](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/StreamingVadSegmenter.ts#L50)
