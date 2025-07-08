[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / VadEvent

# Interface: VadEvent

[realtime-transcription](../modules/realtime_transcription.md).VadEvent

## Table of contents

### Properties

- [analysis](realtime_transcription.VadEvent.md#analysis)
- [confidence](realtime_transcription.VadEvent.md#confidence)
- [currentThreshold](realtime_transcription.VadEvent.md#currentthreshold)
- [duration](realtime_transcription.VadEvent.md#duration)
- [environmentNoise](realtime_transcription.VadEvent.md#environmentnoise)
- [lastSpeechDetectedTime](realtime_transcription.VadEvent.md#lastspeechdetectedtime)
- [sliceIndex](realtime_transcription.VadEvent.md#sliceindex)
- [timestamp](realtime_transcription.VadEvent.md#timestamp)
- [type](realtime_transcription.VadEvent.md#type)

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

[realtime-transcription/types.ts:132](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L132)

___

### confidence

• **confidence**: `number`

#### Defined in

[realtime-transcription/types.ts:127](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L127)

___

### currentThreshold

• `Optional` **currentThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:140](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L140)

___

### duration

• **duration**: `number`

#### Defined in

[realtime-transcription/types.ts:128](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L128)

___

### environmentNoise

• `Optional` **environmentNoise**: `number`

#### Defined in

[realtime-transcription/types.ts:141](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L141)

___

### lastSpeechDetectedTime

• **lastSpeechDetectedTime**: `number`

#### Defined in

[realtime-transcription/types.ts:126](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L126)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:129](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L129)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:125](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L125)

___

### type

• **type**: ``"speech_start"`` \| ``"speech_end"`` \| ``"speech_continue"`` \| ``"silence"``

#### Defined in

[realtime-transcription/types.ts:124](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L124)
