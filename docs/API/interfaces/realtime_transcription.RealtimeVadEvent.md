[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeVadEvent

# Interface: RealtimeVadEvent

[realtime-transcription](../modules/realtime_transcription.md).RealtimeVadEvent

## Table of contents

### Properties

- [analysis](realtime_transcription.RealtimeVadEvent.md#analysis)
- [confidence](realtime_transcription.RealtimeVadEvent.md#confidence)
- [currentThreshold](realtime_transcription.RealtimeVadEvent.md#currentthreshold)
- [duration](realtime_transcription.RealtimeVadEvent.md#duration)
- [environmentNoise](realtime_transcription.RealtimeVadEvent.md#environmentnoise)
- [lastSpeechDetectedTime](realtime_transcription.RealtimeVadEvent.md#lastspeechdetectedtime)
- [sliceIndex](realtime_transcription.RealtimeVadEvent.md#sliceindex)
- [timestamp](realtime_transcription.RealtimeVadEvent.md#timestamp)
- [type](realtime_transcription.RealtimeVadEvent.md#type)

## Properties

### analysis

• `Optional` **analysis**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `averageAmplitude` | `number` |
| `peakAmplitude` | `number` |
| `spectralCentroid?` | `number` |
| `zeroCrossingRate?` | `number` |

#### Defined in

[realtime-transcription/types.ts:126](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L126)

___

### confidence

• **confidence**: `number`

#### Defined in

[realtime-transcription/types.ts:121](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L121)

___

### currentThreshold

• `Optional` **currentThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:134](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L134)

___

### duration

• **duration**: `number`

#### Defined in

[realtime-transcription/types.ts:122](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L122)

___

### environmentNoise

• `Optional` **environmentNoise**: `number`

#### Defined in

[realtime-transcription/types.ts:135](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L135)

___

### lastSpeechDetectedTime

• **lastSpeechDetectedTime**: `number`

#### Defined in

[realtime-transcription/types.ts:120](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L120)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:123](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L123)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:119](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L119)

___

### type

• **type**: ``"speech_start"`` \| ``"speech_end"`` \| ``"speech_continue"`` \| ``"silence"``

#### Defined in

[realtime-transcription/types.ts:118](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L118)
