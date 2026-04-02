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

[realtime-transcription/types.ts:168](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L168)

___

### isCapturing

• **isCapturing**: `boolean`

#### Defined in

[realtime-transcription/types.ts:169](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L169)

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

[realtime-transcription/types.ts:172](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L172)

___

### processTime

• **processTime**: `number`

#### Defined in

[realtime-transcription/types.ts:170](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L170)

___

### recordingTime

• **recordingTime**: `number`

#### Defined in

[realtime-transcription/types.ts:171](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L171)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:167](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L167)

___

### type

• **type**: ``"error"`` \| ``"start"`` \| ``"transcribe"`` \| ``"end"``

#### Defined in

[realtime-transcription/types.ts:166](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L166)

___

### vadEvent

• `Optional` **vadEvent**: [`RealtimeVadEvent`](realtime_transcription.RealtimeVadEvent.md)

#### Defined in

[realtime-transcription/types.ts:177](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L177)
