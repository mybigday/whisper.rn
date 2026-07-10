[whisper.rn](../README.md) / index

# Module: index

## Table of contents

### Classes

- [ParakeetContext](../classes/index.ParakeetContext.md)
- [WhisperContext](../classes/index.WhisperContext.md)
- [WhisperVadContext](../classes/index.WhisperVadContext.md)

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

[index.ts:230](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L230)

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
| `useFlashAttn?` | `boolean` | Use Flash Attention, only recommended if GPU available |
| `useGpu?` | `boolean` | Use GPU if available. Currently iOS only, if it's enabled, Core ML option will be ignored. |

#### Defined in

[index.ts:382](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L382)

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

[index.ts:459](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L459)

___

### ParakeetTranscribeOptions

Ƭ **ParakeetTranscribeOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `audioCtx?` | `number` | Override the model audio context size (0 uses the model default). |
| `maxThreads?` | `number` | Number of threads to use during computation. |

#### Defined in

[index.ts:467](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L467)

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

[index.ts:216](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L216)

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

[NativeRNWhisper.ts:5](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/NativeRNWhisper.ts#L5)

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

[NativeRNWhisper.ts:40](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/NativeRNWhisper.ts#L40)

___

### VadContextOptions

Ƭ **VadContextOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `filePath` | `string` \| `number` | - |
| `isBundleAsset?` | `boolean` | Is the file path a bundle asset for pure string filePath |
| `nThreads?` | `number` | Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores) |
| `useGpu?` | `boolean` | Use GPU if available. Currently iOS only |

#### Defined in

[index.ts:587](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L587)

___

### VadOptions

Ƭ **VadOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `maxSpeechDurationS?` | `number` | Max duration of a speech segment before forcing a new segment in seconds (Default: 30) |
| `minSilenceDurationMs?` | `number` | Min silence duration to consider speech as ended in ms (Default: 100) |
| `minSpeechDurationMs?` | `number` | Min duration for a valid speech segment in ms (Default: 250) |
| `samplesOverlap?` | `number` | Overlap in seconds when copying audio samples from speech segment (Default: 0.1) |
| `speechPadMs?` | `number` | Padding added before and after speech segments in ms (Default: 30) |
| `threshold?` | `number` | Probability threshold to consider as speech (Default: 0.5) |

#### Defined in

[NativeRNWhisper.ts:85](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/NativeRNWhisper.ts#L85)

___

### VadSegment

Ƭ **VadSegment**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `t0` | `number` |
| `t1` | `number` |

#### Defined in

[NativeRNWhisper.ts:113](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/NativeRNWhisper.ts#L113)

## Variables

### isCoreMLAllowFallback

• `Const` **isCoreMLAllowFallback**: `boolean` = `!!nativeConstants.coreMLAllowFallback`

Is allow fallback to CPU if load CoreML model failed

#### Defined in

[index.ts:452](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L452)

___

### isUseCoreML

• `Const` **isUseCoreML**: `boolean` = `!!nativeConstants.useCoreML`

Is use CoreML models on iOS

#### Defined in

[index.ts:449](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L449)

___

### libVersion

• `Const` **libVersion**: `string` = `version`

Current version of whisper.cpp

#### Defined in

[index.ts:446](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L446)

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

[index.ts:715](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L715)

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

[index.ts:550](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L550)

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

[index.ts:405](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L405)

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

[index.ts:663](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L663)

___

### installJsi

▸ **installJsi**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:93](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L93)

___

### releaseAllParakeet

▸ **releaseAllParakeet**(): `Promise`<`void`\>

Release every Parakeet context and free its memory.

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:577](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L577)

___

### releaseAllWhisper

▸ **releaseAllWhisper**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:439](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L439)

___

### releaseAllWhisperVad

▸ **releaseAllWhisperVad**(): `Promise`<`void`\>

Release all VAD contexts and free their memory

#### Returns

`Promise`<`void`\>

Promise resolving when all contexts are released

#### Defined in

[index.ts:695](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L695)

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

[index.ts:704](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L704)
