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

[realtime-transcription/types.ts:130](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L130)

___

### confidence

• **confidence**: `number`

#### Defined in

[realtime-transcription/types.ts:125](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L125)

___

### currentThreshold

• `Optional` **currentThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:138](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L138)

___

### duration

• **duration**: `number`

#### Defined in

[realtime-transcription/types.ts:126](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L126)

___

### environmentNoise

• `Optional` **environmentNoise**: `number`

#### Defined in

[realtime-transcription/types.ts:139](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L139)

___

### lastSpeechDetectedTime

• **lastSpeechDetectedTime**: `number`

#### Defined in

[realtime-transcription/types.ts:124](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L124)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:127](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L127)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:123](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L123)

___

### type

• **type**: ``"speech_start"`` \| ``"speech_end"`` \| ``"speech_continue"`` \| ``"silence"``

#### Defined in

[realtime-transcription/types.ts:122](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L122)
