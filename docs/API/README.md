whisper.rn

# whisper.rn

## Table of contents

### Enumerations

- [AudioSessionCategoryIos](enums/AudioSessionCategoryIos.md)
- [AudioSessionCategoryOptionIos](enums/AudioSessionCategoryOptionIos.md)
- [AudioSessionModeIos](enums/AudioSessionModeIos.md)

### Classes

- [WhisperContext](classes/WhisperContext.md)

### Type Aliases

- [AudioSessionSettingIos](README.md#audiosessionsettingios)
- [ContextOptions](README.md#contextoptions)
- [TranscribeFileOptions](README.md#transcribefileoptions)
- [TranscribeNewSegmentsNativeEvent](README.md#transcribenewsegmentsnativeevent)
- [TranscribeNewSegmentsResult](README.md#transcribenewsegmentsresult)
- [TranscribeOptions](README.md#transcribeoptions)
- [TranscribeProgressNativeEvent](README.md#transcribeprogressnativeevent)
- [TranscribeRealtimeEvent](README.md#transcriberealtimeevent)
- [TranscribeRealtimeNativeEvent](README.md#transcriberealtimenativeevent)
- [TranscribeRealtimeNativePayload](README.md#transcriberealtimenativepayload)
- [TranscribeRealtimeOptions](README.md#transcriberealtimeoptions)
- [TranscribeResult](README.md#transcriberesult)

### Variables

- [AudioSessionIos](README.md#audiosessionios)
- [isCoreMLAllowFallback](README.md#iscoremlallowfallback)
- [isUseCoreML](README.md#isusecoreml)
- [libVersion](README.md#libversion)

### Functions

- [initWhisper](README.md#initwhisper)
- [releaseAllWhisper](README.md#releaseallwhisper)

## Type Aliases

### AudioSessionSettingIos

Ƭ **AudioSessionSettingIos**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `active?` | `boolean` |
| `category` | [`AudioSessionCategoryIos`](enums/AudioSessionCategoryIos.md) |
| `mode?` | [`AudioSessionModeIos`](enums/AudioSessionModeIos.md) |
| `options?` | [`AudioSessionCategoryOptionIos`](enums/AudioSessionCategoryOptionIos.md)[] |

#### Defined in

[index.ts:76](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L76)

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
| `useGpu?` | `boolean` | [Currently iOS only] Use GPU if available. |

#### Defined in

[index.ts:438](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L438)

___

### TranscribeFileOptions

Ƭ **TranscribeFileOptions**: [`TranscribeOptions`](README.md#transcribeoptions) & { `onNewSegments?`: (`result`: [`TranscribeNewSegmentsResult`](README.md#transcribenewsegmentsresult)) => `void` ; `onProgress?`: (`progress`: `number`) => `void`  }

#### Defined in

[index.ts:59](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L59)

___

### TranscribeNewSegmentsNativeEvent

Ƭ **TranscribeNewSegmentsNativeEvent**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `contextId` | `number` |
| `jobId` | `number` |
| `result` | [`TranscribeNewSegmentsResult`](README.md#transcribenewsegmentsresult) |

#### Defined in

[index.ts:52](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L52)

___

### TranscribeNewSegmentsResult

Ƭ **TranscribeNewSegmentsResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `nNew` | `number` |
| `result` | `string` |
| `segments` | [`TranscribeResult`](README.md#transcriberesult)[``"segments"``] |
| `totalNNew` | `number` |

#### Defined in

[index.ts:45](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L45)

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
| `speedUp?` | `boolean` | Speed up audio by x2 (reduced accuracy) |
| `temperature?` | `number` | Tnitial decoding temperature |
| `temperatureInc?` | `number` | - |
| `tokenTimestamps?` | `boolean` | Enable token-level timestamps |
| `translate?` | `boolean` | Translate from source language to english (Default: false) |
| `wordThold?` | `number` | Word timestamp probability threshold |

#### Defined in

[NativeRNWhisper.ts:5](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/NativeRNWhisper.ts#L5)

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

[index.ts:70](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L70)

___

### TranscribeRealtimeEvent

Ƭ **TranscribeRealtimeEvent**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `code` | `number` | - |
| `contextId` | `number` | - |
| `data?` | [`TranscribeResult`](README.md#transcriberesult) | - |
| `error?` | `string` | - |
| `isCapturing` | `boolean` | Is capturing audio, when false, the event is the final result |
| `isStoppedByAction?` | `boolean` | - |
| `jobId` | `number` | - |
| `processTime` | `number` | - |
| `recordingTime` | `number` | - |
| `slices?` | { `code`: `number` ; `data?`: [`TranscribeResult`](README.md#transcriberesult) ; `error?`: `string` ; `processTime`: `number` ; `recordingTime`: `number`  }[] | - |

#### Defined in

[index.ts:133](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L133)

___

### TranscribeRealtimeNativeEvent

Ƭ **TranscribeRealtimeNativeEvent**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `contextId` | `number` |
| `jobId` | `number` |
| `payload` | [`TranscribeRealtimeNativePayload`](README.md#transcriberealtimenativepayload) |

#### Defined in

[index.ts:166](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L166)

___

### TranscribeRealtimeNativePayload

Ƭ **TranscribeRealtimeNativePayload**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `code` | `number` | - |
| `data?` | [`TranscribeResult`](README.md#transcriberesult) | - |
| `error?` | `string` | - |
| `isCapturing` | `boolean` | Is capturing audio, when false, the event is the final result |
| `isStoppedByAction?` | `boolean` | - |
| `isUseSlices` | `boolean` | - |
| `processTime` | `number` | - |
| `recordingTime` | `number` | - |
| `sliceIndex` | `number` | - |

#### Defined in

[index.ts:153](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L153)

___

### TranscribeRealtimeOptions

Ƭ **TranscribeRealtimeOptions**: [`TranscribeOptions`](README.md#transcribeoptions) & { `audioOutputPath?`: `string` ; `audioSessionOnStartIos?`: [`AudioSessionSettingIos`](README.md#audiosessionsettingios) ; `audioSessionOnStopIos?`: `string` \| [`AudioSessionSettingIos`](README.md#audiosessionsettingios) ; `realtimeAudioSec?`: `number` ; `realtimeAudioSliceSec?`: `number` ; `useVad?`: `boolean` ; `vadFreqThold?`: `number` ; `vadMs?`: `number` ; `vadThold?`: `number`  }

#### Defined in

[index.ts:84](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L84)

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

[NativeRNWhisper.ts:37](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/NativeRNWhisper.ts#L37)

## Variables

### AudioSessionIos

• **AudioSessionIos**: `Object`

AudioSession Utility, iOS only.

#### Type declaration

| Name | Type |
| :------ | :------ |
| `Category` | typeof [`AudioSessionCategoryIos`](enums/AudioSessionCategoryIos.md) |
| `CategoryOption` | typeof [`AudioSessionCategoryOptionIos`](enums/AudioSessionCategoryOptionIos.md) |
| `Mode` | typeof [`AudioSessionModeIos`](enums/AudioSessionModeIos.md) |
| `getCurrentCategory` | () => `Promise`<{ `category`: [`AudioSessionCategoryIos`](enums/AudioSessionCategoryIos.md) ; `options`: [`AudioSessionCategoryOptionIos`](enums/AudioSessionCategoryOptionIos.md)[]  }\> |
| `getCurrentMode` | () => `Promise`<[`AudioSessionModeIos`](enums/AudioSessionModeIos.md)\> |
| `setActive` | (`active`: `boolean`) => `Promise`<`void`\> |
| `setCategory` | (`category`: [`AudioSessionCategoryIos`](enums/AudioSessionCategoryIos.md), `options`: [`AudioSessionCategoryOptionIos`](enums/AudioSessionCategoryOptionIos.md)[]) => `Promise`<`void`\> |
| `setMode` | (`mode`: [`AudioSessionModeIos`](enums/AudioSessionModeIos.md)) => `Promise`<`void`\> |

#### Defined in

[AudioSessionIos.ts:50](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/AudioSessionIos.ts#L50)

___

### isCoreMLAllowFallback

• `Const` **isCoreMLAllowFallback**: `boolean` = `!!coreMLAllowFallback`

Is allow fallback to CPU if load CoreML model failed

#### Defined in

[index.ts:540](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L540)

___

### isUseCoreML

• `Const` **isUseCoreML**: `boolean` = `!!useCoreML`

Is use CoreML models on iOS

#### Defined in

[index.ts:537](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L537)

___

### libVersion

• `Const` **libVersion**: `string` = `version`

Current version of whisper.cpp

#### Defined in

[index.ts:532](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L532)

## Functions

### initWhisper

▸ **initWhisper**(`«destructured»`): `Promise`<[`WhisperContext`](classes/WhisperContext.md)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | [`ContextOptions`](README.md#contextoptions) |

#### Returns

`Promise`<[`WhisperContext`](classes/WhisperContext.md)\>

#### Defined in

[index.ts:464](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L464)

___

### releaseAllWhisper

▸ **releaseAllWhisper**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:527](https://github.com/mybigday/whisper.rn/blob/e6c445e/src/index.ts#L527)
