[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeTranscribeEvent

# Interface: RealtimeTranscribeEvent

[realtime-transcription](../modules/realtime_transcription.md).RealtimeTranscribeEvent

## Table of contents

### Properties

- [data](realtime_transcription.RealtimeTranscribeEvent.md#data)
- [endReason](realtime_transcription.RealtimeTranscribeEvent.md#endreason)
- [endSample](realtime_transcription.RealtimeTranscribeEvent.md#endsample)
- [isCapturing](realtime_transcription.RealtimeTranscribeEvent.md#iscapturing)
- [isFinal](realtime_transcription.RealtimeTranscribeEvent.md#isfinal)
- [memoryUsage](realtime_transcription.RealtimeTranscribeEvent.md#memoryusage)
- [processTime](realtime_transcription.RealtimeTranscribeEvent.md#processtime)
- [recordingTime](realtime_transcription.RealtimeTranscribeEvent.md#recordingtime)
- [sliceIndex](realtime_transcription.RealtimeTranscribeEvent.md#sliceindex)
- [startSample](realtime_transcription.RealtimeTranscribeEvent.md#startsample)
- [type](realtime_transcription.RealtimeTranscribeEvent.md#type)
- [vadEvent](realtime_transcription.RealtimeTranscribeEvent.md#vadevent)

## Properties

### data

• `Optional` **data**: [`TranscribeResult`](../modules/index.md#transcriberesult)

#### Defined in

[realtime-transcription/types.ts:189](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L189)

___

### endReason

• `Optional` **endReason**: [`RealtimeSliceEndReason`](../modules/realtime_transcription.md#realtimesliceendreason)

Final transcriptions: why the slice ended. `max_duration` means the speech
went on: the text likely ends mid-sentence (Whisper may still end it with
a period). Undefined when a custom VAD context does not report it.

#### Defined in

[realtime-transcription/types.ts:206](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L206)

___

### endSample

• `Optional` **endSample**: `number`

#### Defined in

[realtime-transcription/types.ts:209](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L209)

___

### isCapturing

• **isCapturing**: `boolean`

#### Defined in

[realtime-transcription/types.ts:190](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L190)

___

### isFinal

• `Optional` **isFinal**: `boolean`

Whether this is the final transcription of the slice (else a draft that a later one replaces)

#### Defined in

[realtime-transcription/types.ts:200](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L200)

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

[realtime-transcription/types.ts:193](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L193)

___

### processTime

• **processTime**: `number`

#### Defined in

[realtime-transcription/types.ts:191](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L191)

___

### recordingTime

• **recordingTime**: `number`

#### Defined in

[realtime-transcription/types.ts:192](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L192)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:188](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L188)

___

### startSample

• `Optional` **startSample**: `number`

Position of the slice's audio in the stream, in samples (`endSample` exclusive)

#### Defined in

[realtime-transcription/types.ts:208](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L208)

___

### type

• **type**: ``"error"`` \| ``"start"`` \| ``"end"`` \| ``"transcribe"``

#### Defined in

[realtime-transcription/types.ts:187](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L187)

___

### vadEvent

• `Optional` **vadEvent**: [`RealtimeVadEvent`](realtime_transcription.RealtimeVadEvent.md)

#### Defined in

[realtime-transcription/types.ts:198](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L198)
