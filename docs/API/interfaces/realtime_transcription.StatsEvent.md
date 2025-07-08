[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / StatsEvent

# Interface: StatsEvent

[realtime-transcription](../modules/realtime_transcription.md).StatsEvent

## Table of contents

### Properties

- [data](realtime_transcription.StatsEvent.md#data)
- [timestamp](realtime_transcription.StatsEvent.md#timestamp)
- [type](realtime_transcription.StatsEvent.md#type)

## Properties

### data

• **data**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `audioStats` | `any` |
| `isActive` | `boolean` |
| `isTranscribing` | `boolean` |
| `queueLength` | `number` |
| `sliceStats` | `any` |
| `vadEnabled` | `boolean` |
| `vadStats` | `any` |

#### Defined in

[realtime-transcription/types.ts:213](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L213)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:206](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L206)

___

### type

• **type**: ``"slice_processed"`` \| ``"vad_change"`` \| ``"queue_change"`` \| ``"memory_change"`` \| ``"status_change"``

#### Defined in

[realtime-transcription/types.ts:207](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L207)
