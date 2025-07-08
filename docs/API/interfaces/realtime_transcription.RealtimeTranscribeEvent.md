[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeTranscribeEvent

# Interface: RealtimeTranscribeEvent

[realtime-transcription](../modules/realtime_transcription.md).RealtimeTranscribeEvent

## Table of contents

### Properties

- [data](realtime_transcription.RealtimeTranscribeEvent.md#data)
- [isCapturing](realtime_transcription.RealtimeTranscribeEvent.md#iscapturing)
- [memoryUsage](realtime_transcription.RealtimeTranscribeEvent.md#memoryusage)
- [processTime](realtime_transcription.RealtimeTranscribeEvent.md#processtime)
- [recordingTime](realtime_transcription.RealtimeTranscribeEvent.md#recordingtime)
- [sliceIndex](realtime_transcription.RealtimeTranscribeEvent.md#sliceindex)
- [type](realtime_transcription.RealtimeTranscribeEvent.md#type)
- [vadEvent](realtime_transcription.RealtimeTranscribeEvent.md#vadevent)

## Properties

### data

• `Optional` **data**: [`TranscribeResult`](../modules/index.md#transcriberesult)

#### Defined in

[realtime-transcription/types.ts:145](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L145)

___

### isCapturing

• **isCapturing**: `boolean`

#### Defined in

[realtime-transcription/types.ts:146](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L146)

___

### memoryUsage

• `Optional` **memoryUsage**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `estimatedMB` | `number` |
| `slicesInMemory` | `number` |
| `totalSamples` | `number` |

#### Defined in

[realtime-transcription/types.ts:149](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L149)

___

### processTime

• **processTime**: `number`

#### Defined in

[realtime-transcription/types.ts:147](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L147)

___

### recordingTime

• **recordingTime**: `number`

#### Defined in

[realtime-transcription/types.ts:148](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L148)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:144](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L144)

___

### type

• **type**: ``"error"`` \| ``"start"`` \| ``"transcribe"`` \| ``"end"``

#### Defined in

[realtime-transcription/types.ts:143](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L143)

___

### vadEvent

• `Optional` **vadEvent**: [`RealtimeVadEvent`](realtime_transcription.RealtimeVadEvent.md)

#### Defined in

[realtime-transcription/types.ts:154](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L154)
