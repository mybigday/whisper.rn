[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeStatsEvent

# Interface: RealtimeStatsEvent

[realtime-transcription](../modules/realtime_transcription.md).RealtimeStatsEvent

## Table of contents

### Properties

- [data](realtime_transcription.RealtimeStatsEvent.md#data)
- [timestamp](realtime_transcription.RealtimeStatsEvent.md#timestamp)
- [type](realtime_transcription.RealtimeStatsEvent.md#type)

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

[realtime-transcription/types.ts:211](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L211)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:204](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L204)

___

### type

• **type**: ``"slice_processed"`` \| ``"vad_change"`` \| ``"queue_change"`` \| ``"memory_change"`` \| ``"status_change"``

#### Defined in

[realtime-transcription/types.ts:205](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L205)
