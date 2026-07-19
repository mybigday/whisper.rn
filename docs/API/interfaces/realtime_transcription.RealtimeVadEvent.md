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

[realtime-transcription/types.ts:158](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L158)

___

### confidence

• **confidence**: `number`

#### Defined in

[realtime-transcription/types.ts:153](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L153)

___

### currentThreshold

• `Optional` **currentThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:166](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L166)

___

### duration

• **duration**: `number`

#### Defined in

[realtime-transcription/types.ts:154](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L154)

___

### environmentNoise

• `Optional` **environmentNoise**: `number`

#### Defined in

[realtime-transcription/types.ts:167](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L167)

___

### lastSpeechDetectedTime

• **lastSpeechDetectedTime**: `number`

#### Defined in

[realtime-transcription/types.ts:152](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L152)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:155](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L155)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:151](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L151)

___

### type

• **type**: ``"speech_start"`` \| ``"speech_end"`` \| ``"speech_continue"`` \| ``"silence"``

#### Defined in

[realtime-transcription/types.ts:150](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L150)
