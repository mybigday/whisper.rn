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

[realtime-transcription/types.ts:149](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L149)

___

### confidence

• **confidence**: `number`

#### Defined in

[realtime-transcription/types.ts:144](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L144)

___

### currentThreshold

• `Optional` **currentThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:157](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L157)

___

### duration

• **duration**: `number`

#### Defined in

[realtime-transcription/types.ts:145](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L145)

___

### environmentNoise

• `Optional` **environmentNoise**: `number`

#### Defined in

[realtime-transcription/types.ts:158](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L158)

___

### lastSpeechDetectedTime

• **lastSpeechDetectedTime**: `number`

#### Defined in

[realtime-transcription/types.ts:143](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L143)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:146](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L146)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:142](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L142)

___

### type

• **type**: ``"speech_start"`` \| ``"speech_end"`` \| ``"speech_continue"`` \| ``"silence"``

#### Defined in

[realtime-transcription/types.ts:141](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L141)
