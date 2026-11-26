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
| `sliceStats` | `any` |
| `vadEnabled` | `boolean` |
| `vadStats` | `any` |

#### Defined in

[realtime-transcription/types.ts:272](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L272)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:266](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L266)

___

### type

• **type**: ``"slice_processed"`` \| ``"vad_change"`` \| ``"memory_change"`` \| ``"status_change"``

#### Defined in

[realtime-transcription/types.ts:267](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L267)
