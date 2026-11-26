[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / ContinuousSpeechSource

# Class: ContinuousSpeechSource

[realtime-transcription](../modules/realtime_transcription.md).ContinuousSpeechSource

A probability source that calls everything speech: `RealtimeTranscriber`
uses it without a VAD, so the segmenter only splits at the maximum segment
duration (at the quietest moment it finds, see `StreamingVadSegmenter`).

## Implements

- [`StreamVadContextLike`](../modules/realtime_transcription.md#streamvadcontextlike)

## Table of contents

### Constructors

- [constructor](realtime_transcription.ContinuousSpeechSource.md#constructor)

### Properties

- [hopSamples](realtime_transcription.ContinuousSpeechSource.md#hopsamples)
- [sampleRate](realtime_transcription.ContinuousSpeechSource.md#samplerate)

### Methods

- [createStream](realtime_transcription.ContinuousSpeechSource.md#createstream)

## Constructors

### constructor

• **new ContinuousSpeechSource**(`sampleRate?`, `hopSamples?`)

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `sampleRate` | `number` | `16000` |
| `hopSamples` | `number` | `512` |

#### Defined in

[realtime-transcription/ContinuousSpeechSource.ts:13](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/ContinuousSpeechSource.ts#L13)

## Properties

### hopSamples

• `Readonly` **hopSamples**: `number`

#### Implementation of

StreamVadContextLike.hopSamples

#### Defined in

[realtime-transcription/ContinuousSpeechSource.ts:9](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/ContinuousSpeechSource.ts#L9)

___

### sampleRate

• `Readonly` **sampleRate**: `number`

#### Implementation of

StreamVadContextLike.sampleRate

#### Defined in

[realtime-transcription/ContinuousSpeechSource.ts:11](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/ContinuousSpeechSource.ts#L11)

## Methods

### createStream

▸ **createStream**(): `Promise`<[`VadStreamLike`](../modules/realtime_transcription.md#vadstreamlike)\>

#### Returns

`Promise`<[`VadStreamLike`](../modules/realtime_transcription.md#vadstreamlike)\>

#### Implementation of

StreamVadContextLike.createStream

#### Defined in

[realtime-transcription/ContinuousSpeechSource.ts:18](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/ContinuousSpeechSource.ts#L18)
