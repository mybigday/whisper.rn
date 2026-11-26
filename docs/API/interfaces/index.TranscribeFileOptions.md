[whisper.rn](../README.md) / [index](../modules/index.md) / TranscribeFileOptions

# Interface: TranscribeFileOptions

[index](../modules/index.md).TranscribeFileOptions

## Hierarchy

- [`TranscribeOptions`](../modules/index.md#transcribeoptions)

  ↳ **`TranscribeFileOptions`**

## Table of contents

### Properties

- [audioCtx](index.TranscribeFileOptions.md#audioctx)
- [beamSize](index.TranscribeFileOptions.md#beamsize)
- [bestOf](index.TranscribeFileOptions.md#bestof)
- [duration](index.TranscribeFileOptions.md#duration)
- [entropyThold](index.TranscribeFileOptions.md#entropythold)
- [includeTokens](index.TranscribeFileOptions.md#includetokens)
- [language](index.TranscribeFileOptions.md#language)
- [logprobThold](index.TranscribeFileOptions.md#logprobthold)
- [maxContext](index.TranscribeFileOptions.md#maxcontext)
- [maxLen](index.TranscribeFileOptions.md#maxlen)
- [maxThreads](index.TranscribeFileOptions.md#maxthreads)
- [maxTokens](index.TranscribeFileOptions.md#maxtokens)
- [nProcessors](index.TranscribeFileOptions.md#nprocessors)
- [noSpeechThold](index.TranscribeFileOptions.md#nospeechthold)
- [offset](index.TranscribeFileOptions.md#offset)
- [onNewSegments](index.TranscribeFileOptions.md#onnewsegments)
- [onProgress](index.TranscribeFileOptions.md#onprogress)
- [prompt](index.TranscribeFileOptions.md#prompt)
- [singleSegment](index.TranscribeFileOptions.md#singlesegment)
- [suppressBlank](index.TranscribeFileOptions.md#suppressblank)
- [suppressNst](index.TranscribeFileOptions.md#suppressnst)
- [tdrzEnable](index.TranscribeFileOptions.md#tdrzenable)
- [temperature](index.TranscribeFileOptions.md#temperature)
- [temperatureInc](index.TranscribeFileOptions.md#temperatureinc)
- [tokenTimestamps](index.TranscribeFileOptions.md#tokentimestamps)
- [translate](index.TranscribeFileOptions.md#translate)
- [wordThold](index.TranscribeFileOptions.md#wordthold)

## Properties

### audioCtx

• `Optional` **audioCtx**: `number` \| ``"auto"``

Encoder audio context size (whisper.cpp `audio_ctx`). A smaller context
encodes less than the model's full 30 s window, which is faster for short
audio at some cost in accuracy.
- number: fixed size, a non-negative integer (0 = model default, the
  default). Must not exceed the model's audio context (1500 for stock models).
- 'auto': fit to the input length (after `offset`/`duration`), with ~2 s
  of headroom, aligned to 64, at least 256 and capped at the model's
  audio context.

Ignored (full context, with a warning) while the Core ML encoder is
active on iOS, whose model has a fixed 30 s input.

#### Inherited from

TranscribeOptions.audioCtx

#### Defined in

[NativeRNWhisper.ts:41](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L41)

___

### beamSize

• `Optional` **beamSize**: `number`

Beam size for beam search

#### Inherited from

TranscribeOptions.beamSize

#### Defined in

[NativeRNWhisper.ts:47](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L47)

___

### bestOf

• `Optional` **bestOf**: `number`

Number of best candidates to keep

#### Inherited from

TranscribeOptions.bestOf

#### Defined in

[NativeRNWhisper.ts:49](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L49)

___

### duration

• `Optional` **duration**: `number`

Duration of audio to process in milliseconds

#### Inherited from

TranscribeOptions.duration

#### Defined in

[NativeRNWhisper.ts:27](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L27)

___

### entropyThold

• `Optional` **entropyThold**: `number`

Retry with a higher temperature when the token entropy is above this, e.g. on repetition (Default: 2.4)

#### Inherited from

TranscribeOptions.entropyThold

#### Defined in

[NativeRNWhisper.ts:62](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L62)

___

### includeTokens

• `Optional` **includeTokens**: `boolean`

Add each segment's text tokens (`segments[].tokens`) to the result (Default: false).
Token times need `tokenTimestamps` (t0 / t1) or a context initialized with DTW (tDtw).

#### Inherited from

TranscribeOptions.includeTokens

#### Defined in

[NativeRNWhisper.ts:73](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L73)

___

### language

• `Optional` **language**: `string`

Spoken language (Default: 'auto' for auto-detect)

#### Inherited from

TranscribeOptions.language

#### Defined in

[NativeRNWhisper.ts:7](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L7)

___

### logprobThold

• `Optional` **logprobThold**: `number`

Retry with a higher temperature when the average log probability is below this (Default: -1)

#### Inherited from

TranscribeOptions.logprobThold

#### Defined in

[NativeRNWhisper.ts:60](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L60)

___

### maxContext

• `Optional` **maxContext**: `number`

Maximum number of text context tokens to store

#### Inherited from

TranscribeOptions.maxContext

#### Defined in

[NativeRNWhisper.ts:15](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L15)

___

### maxLen

• `Optional` **maxLen**: `number`

Maximum segment length in characters

#### Inherited from

TranscribeOptions.maxLen

#### Defined in

[NativeRNWhisper.ts:17](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L17)

___

### maxThreads

• `Optional` **maxThreads**: `number`

Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores)

#### Inherited from

TranscribeOptions.maxThreads

#### Defined in

[NativeRNWhisper.ts:11](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L11)

___

### maxTokens

• `Optional` **maxTokens**: `number`

Maximum number of text tokens per segment, 0 for no limit (Default: 0). A cap proportional to the audio length stops repetition loops early.

#### Inherited from

TranscribeOptions.maxTokens

#### Defined in

[NativeRNWhisper.ts:51](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L51)

___

### nProcessors

• `Optional` **nProcessors**: `number`

Number of processors to use for parallel processing with whisper_full_parallel (Default: 1 to use whisper_full)

#### Inherited from

TranscribeOptions.nProcessors

#### Defined in

[NativeRNWhisper.ts:13](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L13)

___

### noSpeechThold

• `Optional` **noSpeechThold**: `number`

A segment whose no-speech probability is above this, and whose average log
probability is below `logprobThold`, is treated as silence and skipped (Default: 0.6)

#### Inherited from

TranscribeOptions.noSpeechThold

#### Defined in

[NativeRNWhisper.ts:58](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L58)

___

### offset

• `Optional` **offset**: `number`

Time offset in milliseconds

#### Inherited from

TranscribeOptions.offset

#### Defined in

[NativeRNWhisper.ts:25](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L25)

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

[index.ts:234](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L234)

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

[index.ts:232](https://github.com/mybigday/whisper.rn/blob/3a63101/src/index.ts#L232)

___

### prompt

• `Optional` **prompt**: `string`

Initial Prompt

#### Inherited from

TranscribeOptions.prompt

#### Defined in

[NativeRNWhisper.ts:68](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L68)

___

### singleSegment

• `Optional` **singleSegment**: `boolean`

Return the whole input as one segment (Default: false)

#### Inherited from

TranscribeOptions.singleSegment

#### Defined in

[NativeRNWhisper.ts:53](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L53)

___

### suppressBlank

• `Optional` **suppressBlank**: `boolean`

Suppress blank outputs at the beginning of sampling (Default: true)

#### Inherited from

TranscribeOptions.suppressBlank

#### Defined in

[NativeRNWhisper.ts:64](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L64)

___

### suppressNst

• `Optional` **suppressNst**: `boolean`

Suppress non-speech tokens such as [MUSIC] or ♪ (Default: false)

#### Inherited from

TranscribeOptions.suppressNst

#### Defined in

[NativeRNWhisper.ts:66](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L66)

___

### tdrzEnable

• `Optional` **tdrzEnable**: `boolean`

Enable tinydiarize (requires a tdrz model)

#### Inherited from

TranscribeOptions.tdrzEnable

#### Defined in

[NativeRNWhisper.ts:21](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L21)

___

### temperature

• `Optional` **temperature**: `number`

Initial decoding temperature

#### Inherited from

TranscribeOptions.temperature

#### Defined in

[NativeRNWhisper.ts:43](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L43)

___

### temperatureInc

• `Optional` **temperatureInc**: `number`

Temperature fallback increment applied between decoding retries

#### Inherited from

TranscribeOptions.temperatureInc

#### Defined in

[NativeRNWhisper.ts:45](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L45)

___

### tokenTimestamps

• `Optional` **tokenTimestamps**: `boolean`

Enable token-level timestamps

#### Inherited from

TranscribeOptions.tokenTimestamps

#### Defined in

[NativeRNWhisper.ts:19](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L19)

___

### translate

• `Optional` **translate**: `boolean`

Translate from source language to english (Default: false)

#### Inherited from

TranscribeOptions.translate

#### Defined in

[NativeRNWhisper.ts:9](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L9)

___

### wordThold

• `Optional` **wordThold**: `number`

Word timestamp probability threshold

#### Inherited from

TranscribeOptions.wordThold

#### Defined in

[NativeRNWhisper.ts:23](https://github.com/mybigday/whisper.rn/blob/3a63101/src/NativeRNWhisper.ts#L23)
