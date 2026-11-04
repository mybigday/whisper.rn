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
- [ParakeetContextOptions](index.md#parakeetcontextoptions)
- [ParakeetTranscribeOptions](index.md#parakeettranscribeoptions)
- [TranscribeNewSegmentsResult](index.md#transcribenewsegmentsresult)
- [TranscribeOptions](index.md#transcribeoptions)
- [TranscribeResult](index.md#transcriberesult)
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

[index.ts:235](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L235)

___

### ContextOptions

Ƭ **ContextOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `coreMLModelAsset?` | `CoreMLModelAssetOptions` | CoreML model assets, if you're using `require` on filePath, use this option is required if you want to enable Core ML, you will need bundle weights/weight.bin, model.mil, coremldata.bin into app by `require` |
| `filePath` | `string` \| `number` | - |
| `isBundleAsset?` | `boolean` | Is the file path a bundle asset for pure string filePath |
| `useCoreMLIos?` | `boolean` | Prefer to use Core ML model if exists. If set to false, even if the Core ML model exists, it will not be used. |
| `useFlashAttn?` | `boolean` | Use Flash Attention, only recommended if GPU available. Always on when the Hexagon NPU is used. |
| `useGpu?` | `boolean` | Use GPU if available: Metal on iOS, Hexagon NPU on supported Android devices (see the Android installation notes). If it's enabled, Core ML option will be ignored. |

#### Defined in

[index.ts:387](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L387)

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

[index.ts:467](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L467)

___

### ParakeetTranscribeOptions

Ƭ **ParakeetTranscribeOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `audioCtx?` | `number` | Override the model audio context size (0 uses the model default). |
| `maxThreads?` | `number` | Number of threads to use during computation. |

#### Defined in

[index.ts:475](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L475)

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

[index.ts:221](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L221)

___

### TranscribeOptions

Ƭ **TranscribeOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `beamSize?` | `number` | Beam size for beam search |
| `bestOf?` | `number` | Number of best candidates to keep |
| `duration?` | `number` | Duration of audio to process in milliseconds |
| `language?` | `string` | Spoken language (Default: 'auto' for auto-detect) |
| `maxContext?` | `number` | Maximum number of text context tokens to store |
| `maxLen?` | `number` | Maximum segment length in characters |
| `maxThreads?` | `number` | Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores) |
| `nProcessors?` | `number` | Number of processors to use for parallel processing with whisper_full_parallel (Default: 1 to use whisper_full) |
| `offset?` | `number` | Time offset in milliseconds |
| `prompt?` | `string` | Initial Prompt |
| `tdrzEnable?` | `boolean` | Enable tinydiarize (requires a tdrz model) |
| `temperature?` | `number` | Initial decoding temperature |
| `temperatureInc?` | `number` | Temperature fallback increment applied between decoding retries |
| `tokenTimestamps?` | `boolean` | Enable token-level timestamps |
| `translate?` | `boolean` | Translate from source language to english (Default: false) |
| `wordThold?` | `number` | Word timestamp probability threshold |

#### Defined in

[NativeRNWhisper.ts:5](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/NativeRNWhisper.ts#L5)

___

### TranscribeResult

Ƭ **TranscribeResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `isAborted` | `boolean` |
| `language` | `string` |
| `result` | `string` |
| `segments` | { `t0`: `number` ; `t1`: `number` ; `text`: `string`  }[] |

#### Defined in

[NativeRNWhisper.ts:40](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/NativeRNWhisper.ts#L40)

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

[index.ts:595](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L595)

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

[NativeRNWhisper.ts:85](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/NativeRNWhisper.ts#L85)

___

### VadSegment

Ƭ **VadSegment**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `t0` | `number` |
| `t1` | `number` |

#### Defined in

[NativeRNWhisper.ts:127](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/NativeRNWhisper.ts#L127)

## Variables

### isCoreMLAllowFallback

• `Const` **isCoreMLAllowFallback**: `boolean` = `!!nativeConstants.coreMLAllowFallback`

Is allow fallback to CPU if load CoreML model failed

#### Defined in

[index.ts:460](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L460)

___

### isUseCoreML

• `Const` **isUseCoreML**: `boolean` = `!!nativeConstants.useCoreML`

Is use CoreML models on iOS

#### Defined in

[index.ts:457](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L457)

___

### libVersion

• `Const` **libVersion**: `string` = `version`

Current version of whisper.cpp

#### Defined in

[index.ts:454](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L454)

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

[index.ts:831](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L831)

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

[index.ts:558](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L558)

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

[index.ts:413](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L413)

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

[index.ts:779](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L779)

___

### installJsi

▸ **installJsi**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:98](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L98)

___

### releaseAllParakeet

▸ **releaseAllParakeet**(): `Promise`<`void`\>

Release every Parakeet context and free its memory.

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:585](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L585)

___

### releaseAllWhisper

▸ **releaseAllWhisper**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:447](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L447)

___

### releaseAllWhisperVad

▸ **releaseAllWhisperVad**(): `Promise`<`void`\>

Release all VAD contexts and free their memory

#### Returns

`Promise`<`void`\>

Promise resolving when all contexts are released

#### Defined in

[index.ts:811](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L811)

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

[index.ts:820](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L820)
