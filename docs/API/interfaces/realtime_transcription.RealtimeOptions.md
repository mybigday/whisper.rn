[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeOptions

# Interface: RealtimeOptions

[realtime-transcription](../modules/realtime_transcription.md).RealtimeOptions

## Table of contents

### Properties

- [audioMinSec](realtime_transcription.RealtimeOptions.md#audiominsec)
- [audioOutputPath](realtime_transcription.RealtimeOptions.md#audiooutputpath)
- [audioSliceSec](realtime_transcription.RealtimeOptions.md#audioslicesec)
- [audioStreamConfig](realtime_transcription.RealtimeOptions.md#audiostreamconfig)
- [initRealtimeAfterMs](realtime_transcription.RealtimeOptions.md#initrealtimeafterms)
- [initialPrompt](realtime_transcription.RealtimeOptions.md#initialprompt)
- [logger](realtime_transcription.RealtimeOptions.md#logger)
- [maxSlicesInMemory](realtime_transcription.RealtimeOptions.md#maxslicesinmemory)
- [promptPreviousSlices](realtime_transcription.RealtimeOptions.md#promptpreviousslices)
- [realtimeProcessingPauseMs](realtime_transcription.RealtimeOptions.md#realtimeprocessingpausems)
- [transcribeOptions](realtime_transcription.RealtimeOptions.md#transcribeoptions)

## Properties

### audioMinSec

• `Optional` **audioMinSec**: `number`

#### Defined in

[realtime-transcription/types.ts:183](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L183)

___

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:194](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L194)

___

### audioSliceSec

• `Optional` **audioSliceSec**: `number`

#### Defined in

[realtime-transcription/types.ts:182](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L182)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:197](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L197)

___

### initRealtimeAfterMs

• `Optional` **initRealtimeAfterMs**: `number`

#### Defined in

[realtime-transcription/types.ts:204](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L204)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

#### Defined in

[realtime-transcription/types.ts:190](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L190)

___

### logger

• `Optional` **logger**: (`message`: `string`) => `void`

#### Type declaration

▸ (`message`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `message` | `string` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:200](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L200)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

#### Defined in

[realtime-transcription/types.ts:184](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L184)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

#### Defined in

[realtime-transcription/types.ts:191](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L191)

___

### realtimeProcessingPauseMs

• `Optional` **realtimeProcessingPauseMs**: `number`

#### Defined in

[realtime-transcription/types.ts:203](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L203)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeOptions`](../modules/index.md#transcribeoptions)

#### Defined in

[realtime-transcription/types.ts:187](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/types.ts#L187)
