[whisper.rn](../README.md) / index

# Module: index

## Table of contents

### Enumerations

- [AudioSessionCategoryIos](../enums/index.AudioSessionCategoryIos.md)
- [AudioSessionCategoryOptionIos](../enums/index.AudioSessionCategoryOptionIos.md)
- [AudioSessionModeIos](../enums/index.AudioSessionModeIos.md)

### Classes

- [WhisperContext](../classes/index.WhisperContext.md)
- [WhisperVadContext](../classes/index.WhisperVadContext.md)

### Type Aliases

- [AudioSessionSettingIos](index.md#audiosessionsettingios)
- [BenchResult](index.md#benchresult)
- [ContextOptions](index.md#contextoptions)
- [TranscribeFileOptions](index.md#transcribefileoptions)
- [TranscribeNewSegmentsNativeEvent](index.md#transcribenewsegmentsnativeevent)
- [TranscribeNewSegmentsResult](index.md#transcribenewsegmentsresult)
- [TranscribeOptions](index.md#transcribeoptions)
- [TranscribeProgressNativeEvent](index.md#transcribeprogressnativeevent)
- [TranscribeRealtimeEvent](index.md#transcriberealtimeevent)
- [TranscribeRealtimeNativeEvent](index.md#transcriberealtimenativeevent)
- [TranscribeRealtimeNativePayload](index.md#transcriberealtimenativepayload)
- [TranscribeRealtimeOptions](index.md#transcriberealtimeoptions)
- [TranscribeResult](index.md#transcriberesult)
- [VadContextOptions](index.md#vadcontextoptions)
- [VadOptions](index.md#vadoptions)
- [VadSegment](index.md#vadsegment)

### Variables

- [AudioSessionIos](index.md#audiosessionios)
- [isCoreMLAllowFallback](index.md#iscoremlallowfallback)
- [isUseCoreML](index.md#isusecoreml)
- [libVersion](index.md#libversion)

### Functions

- [addNativeLogListener](index.md#addnativeloglistener)
- [initWhisper](index.md#initwhisper)
- [initWhisperVad](index.md#initwhispervad)
- [releaseAllWhisper](index.md#releaseallwhisper)
- [releaseAllWhisperVad](index.md#releaseallwhispervad)
- [toggleNativeLog](index.md#togglenativelog)

## Type Aliases

### AudioSessionSettingIos

Ƭ **AudioSessionSettingIos**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `active?` | `boolean` |
| `category` | [`AudioSessionCategoryIos`](../enums/index.AudioSessionCategoryIos.md) |
| `mode?` | [`AudioSessionModeIos`](../enums/index.AudioSessionModeIos.md) |
| `options?` | [`AudioSessionCategoryOptionIos`](../enums/index.AudioSessionCategoryOptionIos.md)[] |

#### Defined in

[index.ts:135](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L135)

___

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

[index.ts:236](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L236)

___

### ContextOptions

Ƭ **ContextOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `coreMLModelAsset?` | { `assets`: `string`[] \| `number`[] ; `filename`: `string`  } | CoreML model assets, if you're using `require` on filePath, use this option is required if you want to enable Core ML, you will need bundle weights/weight.bin, model.mil, coremldata.bin into app by `require` |
| `coreMLModelAsset.assets` | `string`[] \| `number`[] | - |
| `coreMLModelAsset.filename` | `string` | - |
| `filePath` | `string` \| `number` | - |
| `isBundleAsset?` | `boolean` | Is the file path a bundle asset for pure string filePath |
| `useCoreMLIos?` | `boolean` | Prefer to use Core ML model if exists. If set to false, even if the Core ML model exists, it will not be used. |
| `useFlashAttn?` | `boolean` | Use Flash Attention, only recommended if GPU available |
| `useGpu?` | `boolean` | Use GPU if available. Currently iOS only, if it's enabled, Core ML option will be ignored. |

#### Defined in

[index.ts:616](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L616)

___

### TranscribeFileOptions

Ƭ **TranscribeFileOptions**: [`TranscribeOptions`](index.md#transcribeoptions) & { `onNewSegments?`: (`result`: [`TranscribeNewSegmentsResult`](index.md#transcribenewsegmentsresult)) => `void` ; `onProgress?`: (`progress`: `number`) => `void`  }

#### Defined in

[index.ts:118](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L118)

___

### TranscribeNewSegmentsNativeEvent

Ƭ **TranscribeNewSegmentsNativeEvent**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `contextId` | `number` |
| `jobId` | `number` |
| `result` | [`TranscribeNewSegmentsResult`](index.md#transcribenewsegmentsresult) |

#### Defined in

[index.ts:111](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L111)

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

[index.ts:104](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L104)

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
| `offset?` | `number` | Time offset in milliseconds |
| `prompt?` | `string` | Initial Prompt |
| `tdrzEnable?` | `boolean` | Enable tinydiarize (requires a tdrz model) |
| `temperature?` | `number` | Tnitial decoding temperature |
| `temperatureInc?` | `number` | - |
| `tokenTimestamps?` | `boolean` | Enable token-level timestamps |
| `translate?` | `boolean` | Translate from source language to english (Default: false) |
| `wordThold?` | `number` | Word timestamp probability threshold |

#### Defined in

[NativeRNWhisper.ts:5](https://github.com/mybigday/whisper.rn/blob/0152db5/src/NativeRNWhisper.ts#L5)

___

### TranscribeProgressNativeEvent

Ƭ **TranscribeProgressNativeEvent**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `contextId` | `number` |
| `jobId` | `number` |
| `progress` | `number` |

#### Defined in

[index.ts:129](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L129)

___

### TranscribeRealtimeEvent

Ƭ **TranscribeRealtimeEvent**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `code` | `number` | - |
| `contextId` | `number` | - |
| `data?` | [`TranscribeResult`](index.md#transcriberesult) | - |
| `error?` | `string` | - |
| `isCapturing` | `boolean` | Is capturing audio, when false, the event is the final result |
| `isStoppedByAction?` | `boolean` | - |
| `jobId` | `number` | - |
| `processTime` | `number` | - |
| `recordingTime` | `number` | - |
| `slices?` | { `code`: `number` ; `data?`: [`TranscribeResult`](index.md#transcriberesult) ; `error?`: `string` ; `processTime`: `number` ; `recordingTime`: `number`  }[] | - |

#### Defined in

[index.ts:197](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L197)

___

### TranscribeRealtimeNativeEvent

Ƭ **TranscribeRealtimeNativeEvent**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `contextId` | `number` |
| `jobId` | `number` |
| `payload` | [`TranscribeRealtimeNativePayload`](index.md#transcriberealtimenativepayload) |

#### Defined in

[index.ts:230](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L230)

___

### TranscribeRealtimeNativePayload

Ƭ **TranscribeRealtimeNativePayload**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `code` | `number` | - |
| `data?` | [`TranscribeResult`](index.md#transcriberesult) | - |
| `error?` | `string` | - |
| `isCapturing` | `boolean` | Is capturing audio, when false, the event is the final result |
| `isStoppedByAction?` | `boolean` | - |
| `isUseSlices` | `boolean` | - |
| `processTime` | `number` | - |
| `recordingTime` | `number` | - |
| `sliceIndex` | `number` | - |

#### Defined in

[index.ts:217](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L217)

___

### TranscribeRealtimeOptions

Ƭ **TranscribeRealtimeOptions**: [`TranscribeOptions`](index.md#transcribeoptions) & { `audioOutputPath?`: `string` ; `audioSessionOnStartIos?`: [`AudioSessionSettingIos`](index.md#audiosessionsettingios) ; `audioSessionOnStopIos?`: `string` \| [`AudioSessionSettingIos`](index.md#audiosessionsettingios) ; `realtimeAudioMinSec?`: `number` ; `realtimeAudioSec?`: `number` ; `realtimeAudioSliceSec?`: `number` ; `useVad?`: `boolean` ; `vadFreqThold?`: `number` ; `vadMs?`: `number` ; `vadThold?`: `number`  }

#### Defined in

[index.ts:143](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L143)

___

### TranscribeResult

Ƭ **TranscribeResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `isAborted` | `boolean` |
| `result` | `string` |
| `segments` | { `t0`: `number` ; `t1`: `number` ; `text`: `string`  }[] |

#### Defined in

[NativeRNWhisper.ts:37](https://github.com/mybigday/whisper.rn/blob/0152db5/src/NativeRNWhisper.ts#L37)

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

[index.ts:731](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L731)

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

[NativeRNWhisper.ts:69](https://github.com/mybigday/whisper.rn/blob/0152db5/src/NativeRNWhisper.ts#L69)

___

### VadSegment

Ƭ **VadSegment**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `t0` | `number` |
| `t1` | `number` |

#### Defined in

[NativeRNWhisper.ts:97](https://github.com/mybigday/whisper.rn/blob/0152db5/src/NativeRNWhisper.ts#L97)

## Variables

### AudioSessionIos

• **AudioSessionIos**: `Object`

[Deprecated] AudioSession Utility, iOS only.

#### Type declaration

| Name | Type |
| :------ | :------ |
| `Category` | typeof [`AudioSessionCategoryIos`](../enums/index.AudioSessionCategoryIos.md) |
| `CategoryOption` | typeof [`AudioSessionCategoryOptionIos`](../enums/index.AudioSessionCategoryOptionIos.md) |
| `Mode` | typeof [`AudioSessionModeIos`](../enums/index.AudioSessionModeIos.md) |
| `getCurrentCategory` | () => `Promise`<{ `category`: [`AudioSessionCategoryIos`](../enums/index.AudioSessionCategoryIos.md) ; `options`: [`AudioSessionCategoryOptionIos`](../enums/index.AudioSessionCategoryOptionIos.md)[]  }\> |
| `getCurrentMode` | () => `Promise`<[`AudioSessionModeIos`](../enums/index.AudioSessionModeIos.md)\> |
| `setActive` | (`active`: `boolean`) => `Promise`<`void`\> |
| `setCategory` | (`category`: [`AudioSessionCategoryIos`](../enums/index.AudioSessionCategoryIos.md), `options`: [`AudioSessionCategoryOptionIos`](../enums/index.AudioSessionCategoryOptionIos.md)[]) => `Promise`<`void`\> |
| `setMode` | (`mode`: [`AudioSessionModeIos`](../enums/index.AudioSessionModeIos.md)) => `Promise`<`void`\> |

#### Defined in

[AudioSessionIos.ts:51](https://github.com/mybigday/whisper.rn/blob/0152db5/src/AudioSessionIos.ts#L51)

___

### isCoreMLAllowFallback

• `Const` **isCoreMLAllowFallback**: `boolean` = `!!coreMLAllowFallback`

Is allow fallback to CPU if load CoreML model failed

#### Defined in

[index.ts:723](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L723)

___

### isUseCoreML

• `Const` **isUseCoreML**: `boolean` = `!!useCoreML`

Is use CoreML models on iOS

#### Defined in

[index.ts:720](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L720)

___

### libVersion

• `Const` **libVersion**: `string` = `version`

Current version of whisper.cpp

#### Defined in

[index.ts:715](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L715)

## Functions

### addNativeLogListener

▸ **addNativeLogListener**(`listener`): `Object`

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

[index.ts:869](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L869)

___

### initWhisper

▸ **initWhisper**(`«destructured»`): `Promise`<[`WhisperContext`](../classes/index.WhisperContext.md)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | [`ContextOptions`](index.md#contextoptions) |

#### Returns

`Promise`<[`WhisperContext`](../classes/index.WhisperContext.md)\>

#### Defined in

[index.ts:644](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L644)

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

[index.ts:824](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L824)

___

### releaseAllWhisper

▸ **releaseAllWhisper**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:710](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L710)

___

### releaseAllWhisperVad

▸ **releaseAllWhisperVad**(): `Promise`<`void`\>

Release all VAD contexts and free their memory

#### Returns

`Promise`<`void`\>

Promise resolving when all contexts are released

#### Defined in

[index.ts:861](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L861)

___

### toggleNativeLog

▸ **toggleNativeLog**(`enabled`): `Promise`<`void`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `enabled` | `boolean` |

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:865](https://github.com/mybigday/whisper.rn/blob/0152db5/src/index.ts#L865)
