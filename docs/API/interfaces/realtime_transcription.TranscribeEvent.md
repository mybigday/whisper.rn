[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / TranscribeEvent

# Interface: TranscribeEvent

[realtime-transcription](../modules/realtime_transcription.md).TranscribeEvent

## Table of contents

### Properties

- [data](realtime_transcription.TranscribeEvent.md#data)
- [isCapturing](realtime_transcription.TranscribeEvent.md#iscapturing)
- [memoryUsage](realtime_transcription.TranscribeEvent.md#memoryusage)
- [processTime](realtime_transcription.TranscribeEvent.md#processtime)
- [recordingTime](realtime_transcription.TranscribeEvent.md#recordingtime)
- [sliceIndex](realtime_transcription.TranscribeEvent.md#sliceindex)
- [type](realtime_transcription.TranscribeEvent.md#type)
- [vadEvent](realtime_transcription.TranscribeEvent.md#vadevent)

## Properties

### data

• `Optional` **data**: [`TranscribeResult`](../modules/index.md#transcriberesult)

#### Defined in

[realtime-transcription/types.ts:145](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L145)

___

### isCapturing

• **isCapturing**: `boolean`

#### Defined in

[realtime-transcription/types.ts:146](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L146)

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

[realtime-transcription/types.ts:149](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L149)

___

### processTime

• **processTime**: `number`

#### Defined in

[realtime-transcription/types.ts:147](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L147)

___

### recordingTime

• **recordingTime**: `number`

#### Defined in

[realtime-transcription/types.ts:148](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L148)

___

### sliceIndex

• **sliceIndex**: `number`

#### Defined in

[realtime-transcription/types.ts:144](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L144)

___

### type

• **type**: ``"error"`` \| ``"start"`` \| ``"transcribe"`` \| ``"end"``

#### Defined in

[realtime-transcription/types.ts:143](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L143)

___

### vadEvent

• `Optional` **vadEvent**: [`VadEvent`](realtime_transcription.VadEvent.md)

#### Defined in

[realtime-transcription/types.ts:154](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L154)
