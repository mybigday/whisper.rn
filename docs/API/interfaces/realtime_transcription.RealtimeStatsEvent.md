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

[realtime-transcription/types.ts:238](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L238)

___

### timestamp

• **timestamp**: `number`

#### Defined in

[realtime-transcription/types.ts:232](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L232)

___

### type

• **type**: ``"slice_processed"`` \| ``"vad_change"`` \| ``"memory_change"`` \| ``"status_change"``

#### Defined in

[realtime-transcription/types.ts:233](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L233)
