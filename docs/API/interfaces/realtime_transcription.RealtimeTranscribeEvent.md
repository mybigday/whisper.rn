[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeTranscribeEvent

# Interface: RealtimeTranscribeEvent

[realtime-transcription](../modules/realtime_transcription.md).RealtimeTranscribeEvent

## Table of contents

### Properties

- [data](realtime_transcription.RealtimeTranscribeEvent.md#data)
- [endReason](realtime_transcription.RealtimeTranscribeEvent.md#endreason)
- [isCapturing](realtime_transcription.RealtimeTranscribeEvent.md#iscapturing)
- [isFinal](realtime_transcription.RealtimeTranscribeEvent.md#isfinal)
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

[realtime-transcription/types.ts:185](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L185)

___

### endReason

• `Optional` **endReason**: [`RealtimeSliceEndReason`](../modules/realtime_transcription.md#realtimesliceendreason)

Final transcriptions: why the slice ended. `max_duration` means the speech
went on: the text likely ends mid-sentence (Whisper may still end it with
a period). Undefined when a custom VAD context does not report it.

#### Defined in

[realtime-transcription/types.ts:202](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L202)

___

### isCapturing

• **isCapturing**: `boolean`

#### Defined in

[realtime-transcription/types.ts:186](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L186)

___

### isFinal

• `Optional` **isFinal**: `boolean`

Whether this is the final transcription of the slice (else a draft that a later one replaces)

#### Defined in

[realtime-transcription/types.ts:196](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L196)

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

[realtime-transcription/types.ts:189](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L189)

___

### processTime

• **processTime**: `number`

#### Defined in

[realtime-transcription/types.ts:187](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L187)

___

### recordingTime

• **recordingTime**: `number`

#### Defined in

[realtime-transcription/types.ts:188](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L188)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:184](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L184)

___

### type

• **type**: ``"error"`` \| ``"start"`` \| ``"end"`` \| ``"transcribe"``

#### Defined in

[realtime-transcription/types.ts:183](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L183)

___

### vadEvent

• `Optional` **vadEvent**: [`RealtimeVadEvent`](realtime_transcription.RealtimeVadEvent.md)

#### Defined in

[realtime-transcription/types.ts:194](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L194)
