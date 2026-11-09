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
- [maxPromptSlices](realtime_transcription.RealtimeOptions.md#maxpromptslices)
- [maxResultsInMemory](realtime_transcription.RealtimeOptions.md#maxresultsinmemory)
- [maxSlicesInMemory](realtime_transcription.RealtimeOptions.md#maxslicesinmemory)
- [promptPreviousSlices](realtime_transcription.RealtimeOptions.md#promptpreviousslices)
- [realtimeProcessingPauseMs](realtime_transcription.RealtimeOptions.md#realtimeprocessingpausems)
- [transcribeOptions](realtime_transcription.RealtimeOptions.md#transcribeoptions)

## Properties

### audioMinSec

• `Optional` **audioMinSec**: `number`

#### Defined in

[realtime-transcription/types.ts:179](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L179)

___

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:205](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L205)

___

### audioSliceSec

• `Optional` **audioSliceSec**: `number`

#### Defined in

[realtime-transcription/types.ts:178](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L178)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:208](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L208)

___

### initRealtimeAfterMs

• `Optional` **initRealtimeAfterMs**: `number`

#### Defined in

[realtime-transcription/types.ts:215](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L215)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

Initial Whisper prompt. Ignored when using ParakeetContext.

#### Defined in

[realtime-transcription/types.ts:193](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L193)

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

[realtime-transcription/types.ts:211](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L211)

___

### maxPromptSlices

• `Optional` **maxPromptSlices**: `number`

Maximum number of previous slice results appended to the prompt when
`promptPreviousSlices` is enabled (most recent slices win). Whisper only uses the last
~224 prompt tokens anyway, so a small value avoids re-tokenizing an ever-growing prompt.
Default: unlimited (bounded only by `maxResultsInMemory`).

#### Defined in

[realtime-transcription/types.ts:202](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L202)

___

### maxResultsInMemory

• `Optional` **maxResultsInMemory**: `number`

Maximum number of per-slice transcription results (transcript + segments) retained for
`getTranscriptionResults()`. Older results are dropped oldest-first once the limit is
reached, which keeps long-running (24/7) sessions bounded. Default: unlimited.

#### Defined in

[realtime-transcription/types.ts:187](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L187)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

Maximum number of raw audio slices kept in memory. Default: 3.

#### Defined in

[realtime-transcription/types.ts:181](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L181)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

Add previous Whisper results to the next prompt. Ignored for Parakeet. Defaults to true.

#### Defined in

[realtime-transcription/types.ts:195](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L195)

___

### realtimeProcessingPauseMs

• `Optional` **realtimeProcessingPauseMs**: `number`

#### Defined in

[realtime-transcription/types.ts:214](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L214)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeOptions`](../modules/index.md#transcribeoptions) \| [`ParakeetTranscribeOptions`](../modules/index.md#parakeettranscribeoptions)

Options for the selected Whisper or Parakeet context.

#### Defined in

[realtime-transcription/types.ts:190](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L190)
