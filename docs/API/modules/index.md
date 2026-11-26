[whisper.rn](../README.md) / index

# Module: index

## Table of contents

### Classes

- [ParakeetContext](../classes/index.ParakeetContext.md)
- [WhisperContext](../classes/index.WhisperContext.md)
- [WhisperVadContext](../classes/index.WhisperVadContext.md)
- [WhisperVadStream](../classes/index.WhisperVadStream.md)

### Interfaces

- [TranscribeFileOptions](../interfaces/index.TranscribeFileOptions.md)

### Type Aliases

- [BenchResult](index.md#benchresult)
- [ContextOptions](index.md#contextoptions)
- [DtwAheadsPreset](index.md#dtwaheadspreset)
- [ParakeetContextOptions](index.md#parakeetcontextoptions)
- [ParakeetTranscribeOptions](index.md#parakeettranscribeoptions)
- [TranscribeNewSegmentsResult](index.md#transcribenewsegmentsresult)
- [TranscribeOptions](index.md#transcribeoptions)
- [TranscribeResult](index.md#transcriberesult)
- [TranscribeToken](index.md#transcribetoken)
- [VadContextOptions](index.md#vadcontextoptions)
- [VadOptions](index.md#vadoptions)
- [VadSegment](index.md#vadsegment)

### Variables

- [isCoreMLAllowFallback](index.md#iscoremlallowfallback)
- [isUseCoreML](index.md#isusecoreml)
- [libVersion](index.md#libversion)

### Functions

- [addNativeLogListener](index.md#addnativeloglistener)
- [initParakeet](index.md#initparakeet)
- [initWhisper](index.md#initwhisper)
- [initWhisperVad](index.md#initwhispervad)
- [installJsi](index.md#installjsi)
- [releaseAllParakeet](index.md#releaseallparakeet)
- [releaseAllWhisper](index.md#releaseallwhisper)
- [releaseAllWhisperVad](index.md#releaseallwhispervad)
- [toggleNativeLog](index.md#togglenativelog)

## Type Aliases

### BenchResult

Ƭ **BenchResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `batchMs` | `number` |
| `config` | `string` |
| `decodeMs` | `number` |
| `encodeMs` | `number` |
| `nThreads` | `number` |
| `promptMs` | `number` |

#### Defined in

[index.ts:237](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L237)

___

### ContextOptions

Ƭ **ContextOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `coreMLModelAsset?` | `CoreMLModelAssetOptions` | CoreML model assets, if you're using `require` on filePath, use this option is required if you want to enable Core ML, you will need bundle weights/weight.bin, model.mil, coremldata.bin into app by `require` |
| `dtwAheadsPreset?` | [`DtwAheadsPreset`](index.md#dtwaheadspreset) | Compute DTW-aligned token timestamps (`tokens[].tDtw` with `includeTokens`), more accurate than `tokenTimestamps`, with the alignment heads of this model (Default: off). `n-top-most` uses all heads of the top `dtwNTop` text layers, for other models. Not available with flash attention (so not on the Hexagon NPU): see `WhisperContext.dtw`. |
| `dtwNTop?` | `number` | Text layers used by the `n-top-most` preset (Default: all) |
| `filePath` | `string` \| `number` | - |
| `isBundleAsset?` | `boolean` | Is the file path a bundle asset for pure string filePath |
| `useCoreMLIos?` | `boolean` | Prefer to use Core ML model if exists. If set to false, even if the Core ML model exists, it will not be used. |
| `useFlashAttn?` | `boolean` | Use Flash Attention, only recommended if GPU available. Always on when the Hexagon NPU is used. |
| `useGpu?` | `boolean` | Use GPU if available: Metal on iOS, Hexagon NPU on supported Android devices (see the Android installation notes). If it's enabled, Core ML option will be ignored. |

#### Defined in

[index.ts:410](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L410)

___

### DtwAheadsPreset

Ƭ **DtwAheadsPreset**: ``"tiny.en"`` \| ``"tiny"`` \| ``"base.en"`` \| ``"base"`` \| ``"small.en"`` \| ``"small"`` \| ``"medium.en"`` \| ``"medium"`` \| ``"large-v1"`` \| ``"large-v2"`` \| ``"large-v3"`` \| ``"large-v3-turbo"`` \| ``"n-top-most"``

Alignment heads for DTW token timestamps: the model the context loads, or `n-top-most`

#### Defined in

[index.ts:395](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L395)

___

### ParakeetContextOptions

Ƭ **ParakeetContextOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `filePath` | `string` \| `number` | - |
| `isBundleAsset?` | `boolean` | Is the file path a bundle asset for a string filePath. |
| `useGpu?` | `boolean` | Use GPU acceleration if it is available. |

#### Defined in

[index.ts:504](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L504)

___

### ParakeetTranscribeOptions

Ƭ **ParakeetTranscribeOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `audioCtx?` | `number` | Override the model audio context size (0 uses the model default). |
| `maxThreads?` | `number` | Number of threads to use during computation. |

#### Defined in

[index.ts:512](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L512)

___

### TranscribeNewSegmentsResult

Ƭ **TranscribeNewSegmentsResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `nNew` | `number` |
| `result` | `string` |
| `segments` | [`TranscribeResult`](index.md#transcriberesult)[``"segments"``] |
| `totalNNew` | `number` |

#### Defined in

[index.ts:223](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L223)

___

### TranscribeOptions

Ƭ **TranscribeOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `audioCtx?` | `number` \| ``"auto"`` | Encoder audio context size (whisper.cpp `audio_ctx`). A smaller context encodes less than the model's full 30 s window, which is faster for short audio at some cost in accuracy. - number: fixed size, a non-negative integer (0 = model default, the default). Must not exceed the model's audio context (1500 for stock models). - 'auto': fit to the input length (after `offset`/`duration`), with ~2 s of headroom, aligned to 64, at least 256 and capped at the model's audio context. Ignored (full context, with a warning) while the Core ML encoder is active on iOS, whose model has a fixed 30 s input. |
| `beamSize?` | `number` | Beam size for beam search |
| `bestOf?` | `number` | Number of best candidates to keep |
| `duration?` | `number` | Duration of audio to process in milliseconds |
| `entropyThold?` | `number` | Retry with a higher temperature when the token entropy is above this, e.g. on repetition (Default: 2.4) |
| `includeTokens?` | `boolean` | Add each segment's text tokens (`segments[].tokens`) to the result (Default: false). Token times need `tokenTimestamps` (t0 / t1) or a context initialized with DTW (tDtw). |
| `language?` | `string` | Spoken language (Default: 'auto' for auto-detect) |
| `logprobThold?` | `number` | Retry with a higher temperature when the average log probability is below this (Default: -1) |
| `maxContext?` | `number` | Maximum number of text context tokens to store |
| `maxLen?` | `number` | Maximum segment length in characters |
| `maxThreads?` | `number` | Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores) |
| `maxTokens?` | `number` | Maximum number of text tokens per segment, 0 for no limit (Default: 0). A cap proportional to the audio length stops repetition loops early. |
| `nProcessors?` | `number` | Number of processors to use for parallel processing with whisper_full_parallel (Default: 1 to use whisper_full) |
| `noSpeechThold?` | `number` | A segment whose no-speech probability is above this, and whose average log probability is below `logprobThold`, is treated as silence and skipped (Default: 0.6) |
| `offset?` | `number` | Time offset in milliseconds |
| `prompt?` | `string` | Initial Prompt |
| `singleSegment?` | `boolean` | Return the whole input as one segment (Default: false) |
| `suppressBlank?` | `boolean` | Suppress blank outputs at the beginning of sampling (Default: true) |
| `suppressNst?` | `boolean` | Suppress non-speech tokens such as [MUSIC] or ♪ (Default: false) |
| `tdrzEnable?` | `boolean` | Enable tinydiarize (requires a tdrz model) |
| `temperature?` | `number` | Initial decoding temperature |
| `temperatureInc?` | `number` | Temperature fallback increment applied between decoding retries |
| `tokenTimestamps?` | `boolean` | Enable token-level timestamps |
| `translate?` | `boolean` | Translate from source language to english (Default: false) |
| `wordThold?` | `number` | Word timestamp probability threshold |

#### Defined in

[NativeRNWhisper.ts:5](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L5)

___

### TranscribeResult

Ƭ **TranscribeResult**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `isAborted` | `boolean` | - |
| `language` | `string` | - |
| `languageProb?` | `number` | Whisper: probability of the language when it was auto-detected (`language` unset or 'auto') |
| `result` | `string` | - |
| `segments` | { `avgLogprob?`: `number` ; `noSpeechProb?`: `number` ; `t0`: `number` ; `t1`: `number` ; `text`: `string` ; `tokens?`: [`TranscribeToken`](index.md#transcribetoken)[]  }[] | - |

#### Defined in

[NativeRNWhisper.ts:96](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L96)

___

### TranscribeToken

Ƭ **TranscribeToken**: `Object`

A text token of a Whisper result (special and timestamp tokens are left out)

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `id` | `number` | - |
| `p` | `number` | Probability of the token |
| `plog` | `number` | Log probability of the token |
| `t0` | `number` | Start / end in 10 ms units with `tokenTimestamps`, else -1 |
| `t1` | `number` | - |
| `tDtw` | `number` | DTW-aligned time in 10 ms units when the context uses DTW (`dtwAheadsPreset`), else -1 |
| `text` | `string` | Text of the token. A character split across tokens (common in CJK) comes with the token that completes it, earlier ones have empty text; the texts of a segment's tokens add up to its text. |

#### Defined in

[NativeRNWhisper.ts:77](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L77)

___

### VadContextOptions

Ƭ **VadContextOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `filePath` | `string` \| `number` | Speech model file: a vad.cpp GGUF (Silero, FireRedVAD) or whisper.cpp's `ggml-silero-*.bin` |
| `isBundleAsset?` | `boolean` | Is the file path a bundle asset for pure string filePath |
| `nThreads?` | `number` | Number of threads to use during computation (Default: 1, best for the small per-hop graphs) |
| `useGpu?` | `boolean` | **`Deprecated`** Ignored. VAD always runs on the CPU (vad.cpp's kernels), which is fastest for a single stream. |

#### Defined in

[index.ts:632](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L632)

___

### VadOptions

Ƭ **VadOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `maxSpeechDurationS?` | `number` | Max duration of a speech segment before forcing a new segment in seconds (Default: no limit) |
| `minSilenceDurationMs?` | `number` | Min silence duration to consider speech as ended in ms (Default: 100) |
| `minSpeechDurationMs?` | `number` | Min duration for a valid speech segment in ms (Default: 250) |
| `negThreshold?` | `number` | Probability below which audio counts as silence while in speech, the lower edge of the hysteresis (Default: max(threshold - 0.15, 0.01)) |
| `samplesOverlap?` | `number` | **`Deprecated`** Ignored. It applied to whisper.cpp's windowed VAD; detection now keeps the model state across the whole input, so nothing overlaps. |
| `speechPadMs?` | `number` | Padding added before and after speech segments in ms (Default: 30) |
| `threshold?` | `number` | Probability threshold to consider as speech (Default: 0.5) |

#### Defined in

[NativeRNWhisper.ts:153](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L153)

___

### VadSegment

Ƭ **VadSegment**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `t0` | `number` |
| `t1` | `number` |

#### Defined in

[NativeRNWhisper.ts:195](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L195)

## Variables

### isCoreMLAllowFallback

• `Const` **isCoreMLAllowFallback**: `boolean` = `!!nativeConstants.coreMLAllowFallback`

Is allow fallback to CPU if load CoreML model failed

#### Defined in

[index.ts:497](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L497)

___

### isUseCoreML

• `Const` **isUseCoreML**: `boolean` = `!!nativeConstants.useCoreML`

Is use CoreML models on iOS

#### Defined in

[index.ts:494](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L494)

___

### libVersion

• `Const` **libVersion**: `string` = `version`

Current version of whisper.cpp

#### Defined in

[index.ts:491](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L491)

## Functions

### addNativeLogListener

▸ **addNativeLogListener**(`listener`): `Object`

Add a listener for native whisper.cpp log output

#### Parameters

| Name | Type |
| :------ | :------ |
| `listener` | (`level`: `string`, `text`: `string`) => `void` |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `remove` | () => `void` |

#### Defined in

[index.ts:868](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L868)

___

### initParakeet

▸ **initParakeet**(`«destructured»`): `Promise`<[`ParakeetContext`](../classes/index.ParakeetContext.md)\>

Initialize a Parakeet context with a GGML model file.

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | [`ParakeetContextOptions`](index.md#parakeetcontextoptions) |

#### Returns

`Promise`<[`ParakeetContext`](../classes/index.ParakeetContext.md)\>

#### Defined in

[index.ts:595](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L595)

___

### initWhisper

▸ **initWhisper**(`options`): `Promise`<[`WhisperContext`](../classes/index.WhisperContext.md)\>

Initialize a whisper context with a GGML model file

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `options` | [`ContextOptions`](index.md#contextoptions) | Whisper context options |

#### Returns

`Promise`<[`WhisperContext`](../classes/index.WhisperContext.md)\>

Promise resolving to WhisperContext instance

#### Defined in

[index.ts:446](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L446)

___

### initWhisperVad

▸ **initWhisperVad**(`options`): `Promise`<[`WhisperVadContext`](../classes/index.WhisperVadContext.md)\>

Initialize a VAD context for voice activity detection

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `options` | [`VadContextOptions`](index.md#vadcontextoptions) | VAD context options |

#### Returns

`Promise`<[`WhisperVadContext`](../classes/index.WhisperVadContext.md)\>

Promise resolving to WhisperVadContext instance

#### Defined in

[index.ts:816](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L816)

___

### installJsi

▸ **installJsi**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:99](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L99)

___

### releaseAllParakeet

▸ **releaseAllParakeet**(): `Promise`<`void`\>

Release every Parakeet context and free its memory.

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:622](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L622)

___

### releaseAllWhisper

▸ **releaseAllWhisper**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:484](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L484)

___

### releaseAllWhisperVad

▸ **releaseAllWhisperVad**(): `Promise`<`void`\>

Release all VAD contexts and free their memory

#### Returns

`Promise`<`void`\>

Promise resolving when all contexts are released

#### Defined in

[index.ts:848](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L848)

___

### toggleNativeLog

▸ **toggleNativeLog**(`enabled`): `Promise`<`void`\>

Enable or disable native whisper.cpp logging

#### Parameters

| Name | Type |
| :------ | :------ |
| `enabled` | `boolean` |

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:857](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L857)
