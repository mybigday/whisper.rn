[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeOptions

# Interface: RealtimeOptions

[realtime-transcription](../modules/realtime_transcription.md).RealtimeOptions

## Table of contents

### Properties

- [audioOutputPath](realtime_transcription.RealtimeOptions.md#audiooutputpath)
- [audioStreamConfig](realtime_transcription.RealtimeOptions.md#audiostreamconfig)
- [draftTranscribeOptions](realtime_transcription.RealtimeOptions.md#drafttranscribeoptions)
- [initRealtimeAfterMs](realtime_transcription.RealtimeOptions.md#initrealtimeafterms)
- [initialPrompt](realtime_transcription.RealtimeOptions.md#initialprompt)
- [languageLock](realtime_transcription.RealtimeOptions.md#languagelock)
- [logger](realtime_transcription.RealtimeOptions.md#logger)
- [maxPromptSlices](realtime_transcription.RealtimeOptions.md#maxpromptslices)
- [maxResultsInMemory](realtime_transcription.RealtimeOptions.md#maxresultsinmemory)
- [maxSlicesInMemory](realtime_transcription.RealtimeOptions.md#maxslicesinmemory)
- [minAvgLogprob](realtime_transcription.RealtimeOptions.md#minavglogprob)
- [promptPreviousSlices](realtime_transcription.RealtimeOptions.md#promptpreviousslices)
- [realtimeProcessingPauseMs](realtime_transcription.RealtimeOptions.md#realtimeprocessingpausems)
- [segmenter](realtime_transcription.RealtimeOptions.md#segmenter)
- [splitOverlapMs](realtime_transcription.RealtimeOptions.md#splitoverlapms)
- [transcribeOptions](realtime_transcription.RealtimeOptions.md#transcribeoptions)
- [vadOptions](realtime_transcription.RealtimeOptions.md#vadoptions)
- [vadPreset](realtime_transcription.RealtimeOptions.md#vadpreset)

## Properties

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:278](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L278)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:281](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L281)

___

### draftTranscribeOptions

• `Optional` **draftTranscribeOptions**: `Partial`<[`TranscribeOptions`](../modules/index.md#transcribeoptions)\> \| `Partial`<[`ParakeetTranscribeOptions`](../modules/index.md#parakeettranscribeoptions)\>

Overrides of `transcribeOptions` for drafts (the repeated passes while a
slice grows), e.g. `{ audioCtx: 'auto', maxTokens: 64, singleSegment: true,
temperatureInc: 0 }` to keep them cheap. Default: none.

#### Defined in

[realtime-transcription/types.ts:241](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L241)

___

### initRealtimeAfterMs

• `Optional` **initRealtimeAfterMs**: `number`

Audio a slice needs before its first draft, in ms (Default: 200)

#### Defined in

[realtime-transcription/types.ts:289](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L289)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

Initial Whisper prompt. Ignored when using ParakeetContext.

#### Defined in

[realtime-transcription/types.ts:266](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L266)

___

### languageLock

• `Optional` **languageLock**: `boolean` \| [`LanguageLockOptions`](../modules/realtime_transcription.md#languagelockoptions)

Whisper with language auto-detection: keep the detected language once it
was detected confidently a few times, and pass it explicitly from then on
(see `LanguageLockOptions`). Default: off.

#### Defined in

[realtime-transcription/types.ts:254](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L254)

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

[realtime-transcription/types.ts:284](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L284)

___

### maxPromptSlices

• `Optional` **maxPromptSlices**: `number`

Maximum number of previous slice results appended to the prompt when
`promptPreviousSlices` is enabled (most recent slices win). Whisper only uses the last
~224 prompt tokens anyway, so a small value avoids re-tokenizing an ever-growing prompt.
Default: unlimited (bounded only by `maxResultsInMemory`).

#### Defined in

[realtime-transcription/types.ts:275](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L275)

___

### maxResultsInMemory

• `Optional` **maxResultsInMemory**: `number`

Maximum number of per-slice transcription results (transcript + segments) retained for
`getTranscriptionResults()`. Older results are dropped oldest-first once the limit is
reached, which keeps long-running (24/7) sessions bounded. Default: unlimited.

#### Defined in

[realtime-transcription/types.ts:232](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L232)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

Maximum number of raw audio slices kept in memory. Default: 3.

#### Defined in

[realtime-transcription/types.ts:226](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L226)

___

### minAvgLogprob

• `Optional` **minAvgLogprob**: `number`

Whisper: drop result segments whose average token log probability
(`avgLogprob`) is below this, typically text hallucinated on noise or
silence. Pick it for your model and audio. Default: off.

#### Defined in

[realtime-transcription/types.ts:248](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L248)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

Add previous Whisper results to the next prompt. Ignored for Parakeet. Defaults to true.

#### Defined in

[realtime-transcription/types.ts:268](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L268)

___

### realtimeProcessingPauseMs

• `Optional` **realtimeProcessingPauseMs**: `number`

Audio time between drafts of a slice, in ms (Default: 200)

#### Defined in

[realtime-transcription/types.ts:287](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L287)

___

### segmenter

• `Optional` **segmenter**: [`SpeechSegmenter`](realtime_transcription.SpeechSegmenter.md)

Custom segmentation of the VAD probabilities (Default: `StreamingVadSegmenter`)

#### Defined in

[realtime-transcription/types.ts:224](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L224)

___

### splitOverlapMs

• `Optional` **splitOverlapMs**: `number`

Whisper: audio, in ms, both slices keep beyond a split that falls inside
speech (a max-duration split without a pause). A word cut by the split is
then whole on one side; each word is kept on the side it starts in, by its
token times (DTW times when the context has `dtwAheadsPreset`, else token
timestamps). Final results of such slices are assembled from their tokens.
Default: 0 (off).

#### Defined in

[realtime-transcription/types.ts:263](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L263)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeOptions`](../modules/index.md#transcribeoptions) \| [`ParakeetTranscribeOptions`](../modules/index.md#parakeettranscribeoptions)

Options for the selected Whisper or Parakeet context.

#### Defined in

[realtime-transcription/types.ts:235](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L235)

___

### vadOptions

• `Optional` **vadOptions**: [`VadOptions`](../modules/index.md#vadoptions)

Segmentation options, merged over `vadPreset`. With a `vadContext` they
drive the speech segmentation; without one only `maxSpeechDurationS`
applies (continuous audio is cut into slices of at most that length).
`maxSpeechDurationS` bounds every slice and so the cost of each draft; it
must be finite (Default: the preset's, 30 s for `default`).

#### Defined in

[realtime-transcription/types.ts:220](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L220)

___

### vadPreset

• `Optional` **vadPreset**: ``"continuous"`` \| ``"default"`` \| ``"sensitive"`` \| ``"very-sensitive"`` \| ``"conservative"`` \| ``"very-conservative"`` \| ``"meeting"`` \| ``"noisy"``

Preset for `vadOptions` (Default: 'default')

#### Defined in

[realtime-transcription/types.ts:222](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L222)
