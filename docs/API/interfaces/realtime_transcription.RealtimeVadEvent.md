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

[realtime-transcription/types.ts:152](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L152)

___

### confidence

• **confidence**: `number`

#### Defined in

[realtime-transcription/types.ts:147](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L147)

___

### currentThreshold

• `Optional` **currentThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:160](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L160)

___

### duration

• **duration**: `number`

#### Defined in

[realtime-transcription/types.ts:148](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L148)

___

### environmentNoise

• `Optional` **environmentNoise**: `number`

#### Defined in

[realtime-transcription/types.ts:161](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L161)

___

### lastSpeechDetectedTime

• **lastSpeechDetectedTime**: `number`

#### Defined in

[realtime-transcription/types.ts:146](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L146)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:149](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L149)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:145](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L145)

___

### type

• **type**: ``"speech_start"`` \| ``"speech_end"`` \| ``"speech_continue"`` \| ``"silence"``

#### Defined in

[realtime-transcription/types.ts:144](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L144)
