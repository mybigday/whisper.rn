[whisper.rn](../README.md) / [index](../modules/index.md) / TranscribeFileOptions

# Interface: TranscribeFileOptions

[index](../modules/index.md).TranscribeFileOptions

## Hierarchy

- [`TranscribeOptions`](../modules/index.md#transcribeoptions)

  ↳ **`TranscribeFileOptions`**

## Table of contents

### Properties

- [beamSize](index.TranscribeFileOptions.md#beamsize)
- [bestOf](index.TranscribeFileOptions.md#bestof)
- [duration](index.TranscribeFileOptions.md#duration)
- [language](index.TranscribeFileOptions.md#language)
- [maxContext](index.TranscribeFileOptions.md#maxcontext)
- [maxLen](index.TranscribeFileOptions.md#maxlen)
- [maxThreads](index.TranscribeFileOptions.md#maxthreads)
- [nProcessors](index.TranscribeFileOptions.md#nprocessors)
- [offset](index.TranscribeFileOptions.md#offset)
- [onNewSegments](index.TranscribeFileOptions.md#onnewsegments)
- [onProgress](index.TranscribeFileOptions.md#onprogress)
- [prompt](index.TranscribeFileOptions.md#prompt)
- [tdrzEnable](index.TranscribeFileOptions.md#tdrzenable)
- [temperature](index.TranscribeFileOptions.md#temperature)
- [temperatureInc](index.TranscribeFileOptions.md#temperatureinc)
- [tokenTimestamps](index.TranscribeFileOptions.md#tokentimestamps)
- [translate](index.TranscribeFileOptions.md#translate)
- [wordThold](index.TranscribeFileOptions.md#wordthold)

## Properties

### beamSize

• `Optional` **beamSize**: `number`

Beam size for beam search

#### Inherited from

TranscribeOptions.beamSize

#### Defined in

[NativeRNWhisper.ts:33](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L33)

___

### bestOf

• `Optional` **bestOf**: `number`

Number of best candidates to keep

#### Inherited from

TranscribeOptions.bestOf

#### Defined in

[NativeRNWhisper.ts:35](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L35)

___

### duration

• `Optional` **duration**: `number`

Duration of audio to process in milliseconds

#### Inherited from

TranscribeOptions.duration

#### Defined in

[NativeRNWhisper.ts:27](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L27)

___

### language

• `Optional` **language**: `string`

Spoken language (Default: 'auto' for auto-detect)

#### Inherited from

TranscribeOptions.language

#### Defined in

[NativeRNWhisper.ts:7](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L7)

___

### maxContext

• `Optional` **maxContext**: `number`

Maximum number of text context tokens to store

#### Inherited from

TranscribeOptions.maxContext

#### Defined in

[NativeRNWhisper.ts:15](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L15)

___

### maxLen

• `Optional` **maxLen**: `number`

Maximum segment length in characters

#### Inherited from

TranscribeOptions.maxLen

#### Defined in

[NativeRNWhisper.ts:17](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L17)

___

### maxThreads

• `Optional` **maxThreads**: `number`

Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores)

#### Inherited from

TranscribeOptions.maxThreads

#### Defined in

[NativeRNWhisper.ts:11](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L11)

___

### nProcessors

• `Optional` **nProcessors**: `number`

Number of processors to use for parallel processing with whisper_full_parallel (Default: 1 to use whisper_full)

#### Inherited from

TranscribeOptions.nProcessors

#### Defined in

[NativeRNWhisper.ts:13](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L13)

___

### offset

• `Optional` **offset**: `number`

Time offset in milliseconds

#### Inherited from

TranscribeOptions.offset

#### Defined in

[NativeRNWhisper.ts:25](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L25)

___

### onNewSegments

• `Optional` **onNewSegments**: (`result`: [`TranscribeNewSegmentsResult`](../modules/index.md#transcribenewsegmentsresult)) => `void`

#### Type declaration

▸ (`result`): `void`

Callback when new segments are transcribed

##### Parameters

| Name | Type |
| :------ | :------ |
| `result` | [`TranscribeNewSegmentsResult`](../modules/index.md#transcribenewsegmentsresult) |

##### Returns

`void`

#### Defined in

[index.ts:214](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L214)

___

### onProgress

• `Optional` **onProgress**: (`progress`: `number`) => `void`

#### Type declaration

▸ (`progress`): `void`

Progress callback, the progress is between 0 and 100

##### Parameters

| Name | Type |
| :------ | :------ |
| `progress` | `number` |

##### Returns

`void`

#### Defined in

[index.ts:212](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L212)

___

### prompt

• `Optional` **prompt**: `string`

Initial Prompt

#### Inherited from

TranscribeOptions.prompt

#### Defined in

[NativeRNWhisper.ts:37](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L37)

___

### tdrzEnable

• `Optional` **tdrzEnable**: `boolean`

Enable tinydiarize (requires a tdrz model)

#### Inherited from

TranscribeOptions.tdrzEnable

#### Defined in

[NativeRNWhisper.ts:21](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L21)

___

### temperature

• `Optional` **temperature**: `number`

Initial decoding temperature

#### Inherited from

TranscribeOptions.temperature

#### Defined in

[NativeRNWhisper.ts:29](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L29)

___

### temperatureInc

• `Optional` **temperatureInc**: `number`

Temperature fallback increment applied between decoding retries

#### Inherited from

TranscribeOptions.temperatureInc

#### Defined in

[NativeRNWhisper.ts:31](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L31)

___

### tokenTimestamps

• `Optional` **tokenTimestamps**: `boolean`

Enable token-level timestamps

#### Inherited from

TranscribeOptions.tokenTimestamps

#### Defined in

[NativeRNWhisper.ts:19](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L19)

___

### translate

• `Optional` **translate**: `boolean`

Translate from source language to english (Default: false)

#### Inherited from

TranscribeOptions.translate

#### Defined in

[NativeRNWhisper.ts:9](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L9)

___

### wordThold

• `Optional` **wordThold**: `number`

Word timestamp probability threshold

#### Inherited from

TranscribeOptions.wordThold

#### Defined in

[NativeRNWhisper.ts:23](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/NativeRNWhisper.ts#L23)
