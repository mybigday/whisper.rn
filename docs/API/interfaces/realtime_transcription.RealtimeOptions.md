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

[realtime-transcription/types.ts:208](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L208)

___

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:234](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L234)

___

### audioSliceSec

• `Optional` **audioSliceSec**: `number`

#### Defined in

[realtime-transcription/types.ts:207](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L207)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:237](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L237)

___

### initRealtimeAfterMs

• `Optional` **initRealtimeAfterMs**: `number`

#### Defined in

[realtime-transcription/types.ts:244](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L244)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

Initial Whisper prompt. Ignored when using ParakeetContext.

#### Defined in

[realtime-transcription/types.ts:222](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L222)

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

[realtime-transcription/types.ts:240](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L240)

___

### maxPromptSlices

• `Optional` **maxPromptSlices**: `number`

Maximum number of previous slice results appended to the prompt when
`promptPreviousSlices` is enabled (most recent slices win). Whisper only uses the last
~224 prompt tokens anyway, so a small value avoids re-tokenizing an ever-growing prompt.
Default: unlimited (bounded only by `maxResultsInMemory`).

#### Defined in

[realtime-transcription/types.ts:231](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L231)

___

### maxResultsInMemory

• `Optional` **maxResultsInMemory**: `number`

Maximum number of per-slice transcription results (transcript + segments) retained for
`getTranscriptionResults()`. Older results are dropped oldest-first once the limit is
reached, which keeps long-running (24/7) sessions bounded. Default: unlimited.

#### Defined in

[realtime-transcription/types.ts:216](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L216)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

Maximum number of raw audio slices kept in memory. Default: 3.

#### Defined in

[realtime-transcription/types.ts:210](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L210)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

Add previous Whisper results to the next prompt. Ignored for Parakeet. Defaults to true.

#### Defined in

[realtime-transcription/types.ts:224](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L224)

___

### realtimeProcessingPauseMs

• `Optional` **realtimeProcessingPauseMs**: `number`

#### Defined in

[realtime-transcription/types.ts:243](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L243)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeOptions`](../modules/index.md#transcribeoptions) \| [`ParakeetTranscribeOptions`](../modules/index.md#parakeettranscribeoptions)

Options for the selected Whisper or Parakeet context.

#### Defined in

[realtime-transcription/types.ts:219](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L219)
