whisper.rn

# whisper.rn

## Table of contents

### Classes

- [WhisperContext](classes/WhisperContext.md)

### Type Aliases

- [TranscribeOptions](README.md#transcribeoptions)
- [TranscribeRealtimeEvent](README.md#transcriberealtimeevent)
- [TranscribeRealtimeNativeEvent](README.md#transcriberealtimenativeevent)
- [TranscribeRealtimeNativePayload](README.md#transcriberealtimenativepayload)
- [TranscribeRealtimeOptions](README.md#transcriberealtimeoptions)
- [TranscribeResult](README.md#transcriberesult)

### Variables

- [isCoreMLAllowFallback](README.md#iscoremlallowfallback)
- [isUseCoreML](README.md#isusecoreml)
- [libVersion](README.md#libversion)

### Functions

- [initWhisper](README.md#initwhisper)
- [releaseAllWhisper](README.md#releaseallwhisper)

## Type Aliases

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

[index.ts:36](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L36)

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

[index.ts:92](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L92)

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

[index.ts:125](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L125)

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

[index.ts:112](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L112)

___

### TranscribeRealtimeOptions

Ƭ **TranscribeRealtimeOptions**: [`TranscribeOptions`](README.md#transcribeoptions) & { `realtimeAudioSec?`: `number` ; `realtimeAudioSliceSec?`: `number`  }

#### Defined in

[index.ts:68](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L68)

___

### TranscribeResult

Ƭ **TranscribeResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `result` | `string` |
| `segments` | { `t0`: `number` ; `t1`: `number` ; `text`: `string`  }[] |

#### Defined in

[index.ts:83](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L83)

## Variables

### isCoreMLAllowFallback

• `Const` **isCoreMLAllowFallback**: `boolean` = `!!RNWhisper.WHISPER_COREML_ALLOW_FALLBACK`

Is allow fallback to CPU if load CoreML model failed

#### Defined in

[index.ts:277](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L277)

___

### isUseCoreML

• `Const` **isUseCoreML**: `boolean` = `!!RNWhisper.WHISPER_USE_COREML`

Is use CoreML models on iOS

#### Defined in

[index.ts:274](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L274)

___

### libVersion

• `Const` **libVersion**: `string` = `version`

Current version of whisper.cpp

#### Defined in

[index.ts:271](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L271)

## Functions

### initWhisper

▸ **initWhisper**(`«destructured»?`): `Promise`<[`WhisperContext`](classes/WhisperContext.md)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `Object` |
| › `filePath?` | `string` |
| › `isBundleAsset?` | `boolean` |

#### Returns

`Promise`<[`WhisperContext`](classes/WhisperContext.md)\>

#### Defined in

[index.ts:259](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L259)

___

### releaseAllWhisper

▸ **releaseAllWhisper**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:266](https://github.com/mybigday/whisper.rn/blob/e16f2a4/src/index.ts#L266)
