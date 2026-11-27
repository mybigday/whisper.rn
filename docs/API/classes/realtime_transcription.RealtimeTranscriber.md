[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeTranscriber

# Class: RealtimeTranscriber

[realtime-transcription](../modules/realtime_transcription.md).RealtimeTranscriber

RealtimeTranscriber transcribes a live audio stream in slices.

- The audio goes through a segmenter: each speech segment becomes a slice.
  Given a speech probability source (a `WhisperVadContext`, or "always
  speech" without a VAD) the transcriber segments the probabilities itself
  (`StreamVad`): segments longer than `maxSpeechDurationS` are split, at an
  earlier pause when there is one. A segmenter VAD (`RealtimeSegmenterLike`)
  is used as it is.
- While a slice grows it gets draft transcriptions (`isFinal: false`); when
  it ends, one final transcription (`isFinal: true`, with `endReason`).
- Finals run before drafts, and a running draft is aborted for a final.
- Positions are counted in samples of the audio, not wall-clock time, so a
  recorded stream replays the same way.

## Table of contents

### Constructors

- [constructor](realtime_transcription.RealtimeTranscriber.md#constructor)

### Accessors

- [lockedLanguage](realtime_transcription.RealtimeTranscriber.md#lockedlanguage)

### Methods

- [getStatistics](realtime_transcription.RealtimeTranscriber.md#getstatistics)
- [getTranscriptionResults](realtime_transcription.RealtimeTranscriber.md#gettranscriptionresults)
- [nextSlice](realtime_transcription.RealtimeTranscriber.md#nextslice)
- [release](realtime_transcription.RealtimeTranscriber.md#release)
- [reset](realtime_transcription.RealtimeTranscriber.md#reset)
- [start](realtime_transcription.RealtimeTranscriber.md#start)
- [stop](realtime_transcription.RealtimeTranscriber.md#stop)
- [updateCallbacks](realtime_transcription.RealtimeTranscriber.md#updatecallbacks)
- [updateVadOptions](realtime_transcription.RealtimeTranscriber.md#updatevadoptions)

## Constructors

### constructor

• **new RealtimeTranscriber**(`dependencies`, `options?`, `callbacks?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `dependencies` | [`RealtimeTranscriberDependencies`](../modules/realtime_transcription.md#realtimetranscriberdependencies) |
| `options` | [`RealtimeOptions`](../interfaces/realtime_transcription.RealtimeOptions.md) |
| `callbacks` | [`RealtimeTranscriberCallbacks`](../interfaces/realtime_transcription.RealtimeTranscriberCallbacks.md) |

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:227](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L227)

## Accessors

### lockedLanguage

• `get` **lockedLanguage**(): `undefined` \| `string`

The language the language lock holds, if any

#### Returns

`undefined` \| `string`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1012](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L1012)

## Methods

### getStatistics

▸ **getStatistics**(): `Object`

Get current statistics

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `audioStats` | { `accumulatedSamples`: `number` ; `isRecording`: `boolean`  } |
| `audioStats.accumulatedSamples` | `number` |
| `audioStats.isRecording` | `boolean` |
| `isActive` | `boolean` |
| `isTranscribing` | `boolean` |
| `sliceStats` | { `currentSliceIndex`: `number` ; `memoryUsage`: [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md) ; `totalSlices`: `number` ; `transcribeSliceIndex`: `number`  } |
| `sliceStats.currentSliceIndex` | `number` |
| `sliceStats.memoryUsage` | [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md) |
| `sliceStats.totalSlices` | `number` |
| `sliceStats.transcribeSliceIndex` | `number` |
| `vadEnabled` | `boolean` |
| `vadStats` | ``null`` \| { `contextAvailable`: `boolean` = true; `enabled`: `boolean` = true; `lastSpeechDetectedTime`: `number`  } |

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1019](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L1019)

___

### getTranscriptionResults

▸ **getTranscriptionResults**(): { `slice`: [`AudioSliceNoData`](../interfaces/realtime_transcription.AudioSliceNoData.md) ; `transcribeEvent`: [`RealtimeTranscribeEvent`](../interfaces/realtime_transcription.RealtimeTranscribeEvent.md)  }[]

Get the retained transcription results, oldest first.

When `maxResultsInMemory` is set this is only the most recent window; callers that need
every result should collect them from the `onTranscribe` callback as they arrive.

#### Returns

{ `slice`: [`AudioSliceNoData`](../interfaces/realtime_transcription.AudioSliceNoData.md) ; `transcribeEvent`: [`RealtimeTranscribeEvent`](../interfaces/realtime_transcription.RealtimeTranscribeEvent.md)  }[]

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1045](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L1045)

___

### nextSlice

▸ **nextSlice**(): `Promise`<`void`\>

Finalize the current slice now (`endReason: 'manual'`); speech that goes
on continues in the next slice.

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1056](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L1056)

___

### release

▸ **release**(): `Promise`<`void`\>

Release all resources. The VAD context stays the caller's; only the
stream this transcriber created on it is freed.

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1160](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L1160)

___

### reset

▸ **reset**(): `void`

Reset all components

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1118](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L1118)

___

### start

▸ **start**(): `Promise`<`void`\>

Start realtime transcription

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:346](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L346)

___

### stop

▸ **stop**(): `Promise`<`void`\>

Stop realtime transcription. Speech still open is closed and transcribed
(final, `endReason: 'end_of_stream'`) before this resolves; pending
drafts are dropped.

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:398](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L398)

___

### updateCallbacks

▸ **updateCallbacks**(`callbacks`): `void`

Update callbacks

#### Parameters

| Name | Type |
| :------ | :------ |
| `callbacks` | `Partial`<[`RealtimeTranscriberCallbacks`](../interfaces/realtime_transcription.RealtimeTranscriberCallbacks.md)\> |

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:996](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L996)

___

### updateVadOptions

▸ **updateVadOptions**(`options`): `void`

Update segmentation options dynamically (maxSpeechDurationS stays finite)

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Partial`<[`VadOptions`](../modules/index.md#vadoptions)\> |

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1003](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/RealtimeTranscriber.ts#L1003)
