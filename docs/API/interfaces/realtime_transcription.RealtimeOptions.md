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

[realtime-transcription/types.ts:188](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L188)

___

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:200](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L200)

___

### audioSliceSec

• `Optional` **audioSliceSec**: `number`

#### Defined in

[realtime-transcription/types.ts:187](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L187)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:203](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L203)

___

### initRealtimeAfterMs

• `Optional` **initRealtimeAfterMs**: `number`

#### Defined in

[realtime-transcription/types.ts:210](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L210)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

Initial Whisper prompt. Ignored when using ParakeetContext.

#### Defined in

[realtime-transcription/types.ts:195](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L195)

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

[realtime-transcription/types.ts:206](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L206)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

#### Defined in

[realtime-transcription/types.ts:189](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L189)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

Add previous Whisper results to the next prompt. Ignored for Parakeet. Defaults to true.

#### Defined in

[realtime-transcription/types.ts:197](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L197)

___

### realtimeProcessingPauseMs

• `Optional` **realtimeProcessingPauseMs**: `number`

#### Defined in

[realtime-transcription/types.ts:209](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L209)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeOptions`](../modules/index.md#transcribeoptions) \| [`ParakeetTranscribeOptions`](../modules/index.md#parakeettranscribeoptions)

Options for the selected Whisper or Parakeet context.

#### Defined in

[realtime-transcription/types.ts:192](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/types.ts#L192)
