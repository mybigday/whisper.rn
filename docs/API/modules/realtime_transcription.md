[whisper.rn](../README.md) / realtime-transcription

# Module: realtime-transcription

## Table of contents

### Classes

- [ContinuousSpeechSource](../classes/realtime_transcription.ContinuousSpeechSource.md)
- [LanguageLock](../classes/realtime_transcription.LanguageLock.md)
- [RealtimeTranscriber](../classes/realtime_transcription.RealtimeTranscriber.md)
- [RingBuffer](../classes/realtime_transcription.RingBuffer.md)
- [RingBufferVad](../classes/realtime_transcription.RingBufferVad.md)
- [SliceManager](../classes/realtime_transcription.SliceManager.md)
- [StreamVad](../classes/realtime_transcription.StreamVad.md)
- [StreamingVadSegmenter](../classes/realtime_transcription.StreamingVadSegmenter.md)

### Interfaces

- [AudioSlice](../interfaces/realtime_transcription.AudioSlice.md)
- [AudioSliceNoData](../interfaces/realtime_transcription.AudioSliceNoData.md)
- [AudioStreamConfig](../interfaces/realtime_transcription.AudioStreamConfig.md)
- [AudioStreamData](../interfaces/realtime_transcription.AudioStreamData.md)
- [AudioStreamInterface](../interfaces/realtime_transcription.AudioStreamInterface.md)
- [MemoryUsage](../interfaces/realtime_transcription.MemoryUsage.md)
- [RealtimeOptions](../interfaces/realtime_transcription.RealtimeOptions.md)
- [RealtimeStatsEvent](../interfaces/realtime_transcription.RealtimeStatsEvent.md)
- [RealtimeTranscribeEvent](../interfaces/realtime_transcription.RealtimeTranscribeEvent.md)
- [RealtimeTranscriberCallbacks](../interfaces/realtime_transcription.RealtimeTranscriberCallbacks.md)
- [RealtimeVadContextLike](../interfaces/realtime_transcription.RealtimeVadContextLike.md)
- [RealtimeVadEvent](../interfaces/realtime_transcription.RealtimeVadEvent.md)
- [SpeechSegmenter](../interfaces/realtime_transcription.SpeechSegmenter.md)
- [VadSpeechChunkInfo](../interfaces/realtime_transcription.VadSpeechChunkInfo.md)
- [VadSpeechEndInfo](../interfaces/realtime_transcription.VadSpeechEndInfo.md)
- [WavFileWriterFs](../interfaces/realtime_transcription.WavFileWriterFs.md)

### Type Aliases

- [LanguageLockOptions](realtime_transcription.md#languagelockoptions)
- [RealtimeSliceEndReason](realtime_transcription.md#realtimesliceendreason)
- [RealtimeTranscriberDependencies](realtime_transcription.md#realtimetranscriberdependencies)
- [StreamVadContextLike](realtime_transcription.md#streamvadcontextlike)
- [StreamVadOptions](realtime_transcription.md#streamvadoptions)
- [StreamingVadSegmenterOptions](realtime_transcription.md#streamingvadsegmenteroptions)
- [VadSegmentEndReason](realtime_transcription.md#vadsegmentendreason)
- [VadSegmenterEvent](realtime_transcription.md#vadsegmenterevent)
- [VadStreamLike](realtime_transcription.md#vadstreamlike)

### Variables

- [VAD\_PRESETS](realtime_transcription.md#vad_presets)

## Type Aliases

### LanguageLockOptions

Ƭ **LanguageLockOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `lockAfter?` | `number` | Lock once a language was detected this many times with `minProb` (Default: 3) |
| `minProb?` | `number` | Detections below this probability do not count (Default: 0.5) |
| `recheckEvery?` | `number` | While locked, auto-detect again on every Nth final (Default: 8) |
| `recheckMinMs?` | `number` | Only re-check on finals at least this long, in ms (Default: 3000) |
| `unlockAfter?` | `number` | Unlock after this many re-checks in a row confidently detect another language (Default: 2) |
| `unlockProb?` | `number` | Probability a re-check needs to count against the lock (Default: 0.9) |

#### Defined in

[realtime-transcription/LanguageLock.ts:1](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/LanguageLock.ts#L1)

___

### RealtimeSliceEndReason

Ƭ **RealtimeSliceEndReason**: [`VadSegmentEndReason`](realtime_transcription.md#vadsegmentendreason) \| ``"manual"``

Why a slice was finalized: its VAD segment's end reason, or `manual` (`nextSlice()`)

#### Defined in

[realtime-transcription/types.ts:184](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L184)

___

### RealtimeTranscriberDependencies

Ƭ **RealtimeTranscriberDependencies**: { `audioStream`: [`AudioStreamInterface`](../interfaces/realtime_transcription.AudioStreamInterface.md) ; `fs?`: [`WavFileWriterFs`](../interfaces/realtime_transcription.WavFileWriterFs.md) ; `vadContext?`: [`StreamVadContextLike`](realtime_transcription.md#streamvadcontextlike)  } & { `parakeetContext?`: `never` ; `whisperContext`: `WhisperContextLike`  } \| { `parakeetContext`: `ParakeetContextLike` ; `whisperContext?`: `never`  }

Dependencies for realtime transcription. Provide exactly one transcription
context: `whisperContext` or `parakeetContext`.

#### Defined in

[realtime-transcription/types.ts:408](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L408)

___

### StreamVadContextLike

Ƭ **StreamVadContextLike**: `Object`

The part of `WhisperVadContext` StreamVad uses

#### Type declaration

| Name | Type |
| :------ | :------ |
| `hopSamples` | `number` |
| `sampleRate` | `number` |
| `createStream` | () => `Promise`<[`VadStreamLike`](realtime_transcription.md#vadstreamlike)\> |

#### Defined in

[realtime-transcription/StreamVad.ts:16](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/StreamVad.ts#L16)

___

### StreamVadOptions

Ƭ **StreamVadOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `logger?` | (`message`: `string`) => `void` | - |
| `segmenter?` | [`SpeechSegmenter`](../interfaces/realtime_transcription.SpeechSegmenter.md) | Segmentation of the probabilities (Default: a `StreamingVadSegmenter` with `vadOptions`). A custom one gets `vadOptions` through `setOptions`. |
| `vadOptions?` | [`VadOptions`](index.md#vadoptions) | - |
| `vadPreset?` | keyof typeof [`VAD_PRESETS`](realtime_transcription.md#vad_presets) | - |

#### Defined in

[realtime-transcription/StreamVad.ts:22](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/StreamVad.ts#L22)

___

### StreamingVadSegmenterOptions

Ƭ **StreamingVadSegmenterOptions**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `frameEnergy?` | (`start`: `number`, `end`: `number`) => `number` | Energy (e.g. mean square) of the audio samples `[start, end)`, used to pick the quietest of the near-lowest frames for a dip split. Without it, the latest frame with the lowest probability wins. |
| `splitAtDip?` | `boolean` | Where a segment without any usable pause is split at `maxSpeechDurationS`. - true (default): at the least speech-like frame of the last 3 s (0.5 s before the limit at the latest, and in the second half of the segment), the quietest one per `frameEnergy` when several are about as low. An ongoing pause longer than every earlier one ends the segment where it began. - false: right at the limit, exactly as the offline pass does. |

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:61](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/StreamingVadSegmenter.ts#L61)

___

### VadSegmentEndReason

Ƭ **VadSegmentEndReason**: ``"pause"`` \| ``"max_duration"`` \| ``"end_of_stream"``

Why a speech segment ended

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:4](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/StreamingVadSegmenter.ts#L4)

___

### VadSegmenterEvent

Ƭ **VadSegmenterEvent**: { `prob`: `number` ; `sample`: `number` ; `type`: ``"start"``  } \| { `inSpeech?`: `boolean` ; `prob`: `number` ; `reason`: [`VadSegmentEndReason`](realtime_transcription.md#vadsegmentendreason) ; `sample`: `number` ; `type`: ``"end"``  }

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:12](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/StreamingVadSegmenter.ts#L12)

___

### VadStreamLike

Ƭ **VadStreamLike**: `Object`

The part of `WhisperVadStream` StreamVad uses

#### Type declaration

| Name | Type |
| :------ | :------ |
| `flush` | () => `Promise`<`number`[]\> |
| `push` | (`audio`: `Uint8Array`) => `Promise`<`number`[]\> |
| `release` | () => `Promise`<`void`\> |
| `reset` | () => `Promise`<`void`\> |

#### Defined in

[realtime-transcription/StreamVad.ts:8](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/StreamVad.ts#L8)

## Variables

### VAD\_PRESETS

• `Const` **VAD\_PRESETS**: `Object`

VAD Presets Overview:

                           VAD Presets
                        /      |      \
               Conservative  Default  Sensitive
               /        |        |        \
       conservative  very-conservative  sensitive  very-sensitive
       (0.7 thresh)   (0.8 thresh)    (0.3 thresh) (0.2 thresh)
       500ms min      750ms min       100ms min    100ms min
       Clear speech   Very clear      Quiet env    Catches whispers

                        Specialized Presets
                     /        |        \
               continuous   meeting    noisy
               (60s max)    (45s max)  (0.75 thresh)
               Lectures     Multi-spk   Strict for noise

Key Parameters:
- threshold: 0.0-1.0 (lower = more sensitive)
- minSpeechDurationMs: Min duration to consider speech
- minSilenceDurationMs: Min silence before ending speech
- maxSpeechDurationS: Max continuous speech duration
- speechPadMs: Padding around detected speech

#### Type declaration

| Name | Type |
| :------ | :------ |
| `conservative` | { `maxSpeechDurationS`: `number` = 25; `minSilenceDurationMs`: `number` = 200; `minSpeechDurationMs`: `number` = 500; `speechPadMs`: `number` = 20; `threshold`: `number` = 0.7 } |
| `conservative.maxSpeechDurationS` | `number` |
| `conservative.minSilenceDurationMs` | `number` |
| `conservative.minSpeechDurationMs` | `number` |
| `conservative.speechPadMs` | `number` |
| `conservative.threshold` | `number` |
| `continuous` | { `maxSpeechDurationS`: `number` = 60; `minSilenceDurationMs`: `number` = 300; `minSpeechDurationMs`: `number` = 200; `speechPadMs`: `number` = 50; `threshold`: `number` = 0.4 } |
| `continuous.maxSpeechDurationS` | `number` |
| `continuous.minSilenceDurationMs` | `number` |
| `continuous.minSpeechDurationMs` | `number` |
| `continuous.speechPadMs` | `number` |
| `continuous.threshold` | `number` |
| `default` | { `maxSpeechDurationS`: `number` = 30; `minSilenceDurationMs`: `number` = 100; `minSpeechDurationMs`: `number` = 250; `speechPadMs`: `number` = 30; `threshold`: `number` = 0.5 } |
| `default.maxSpeechDurationS` | `number` |
| `default.minSilenceDurationMs` | `number` |
| `default.minSpeechDurationMs` | `number` |
| `default.speechPadMs` | `number` |
| `default.threshold` | `number` |
| `meeting` | { `maxSpeechDurationS`: `number` = 45; `minSilenceDurationMs`: `number` = 150; `minSpeechDurationMs`: `number` = 300; `speechPadMs`: `number` = 75; `threshold`: `number` = 0.45 } |
| `meeting.maxSpeechDurationS` | `number` |
| `meeting.minSilenceDurationMs` | `number` |
| `meeting.minSpeechDurationMs` | `number` |
| `meeting.speechPadMs` | `number` |
| `meeting.threshold` | `number` |
| `noisy` | { `maxSpeechDurationS`: `number` = 25; `minSilenceDurationMs`: `number` = 100; `minSpeechDurationMs`: `number` = 400; `speechPadMs`: `number` = 40; `threshold`: `number` = 0.75 } |
| `noisy.maxSpeechDurationS` | `number` |
| `noisy.minSilenceDurationMs` | `number` |
| `noisy.minSpeechDurationMs` | `number` |
| `noisy.speechPadMs` | `number` |
| `noisy.threshold` | `number` |
| `sensitive` | { `maxSpeechDurationS`: `number` = 15; `minSilenceDurationMs`: `number` = 50; `minSpeechDurationMs`: `number` = 100; `speechPadMs`: `number` = 50; `threshold`: `number` = 0.3 } |
| `sensitive.maxSpeechDurationS` | `number` |
| `sensitive.minSilenceDurationMs` | `number` |
| `sensitive.minSpeechDurationMs` | `number` |
| `sensitive.speechPadMs` | `number` |
| `sensitive.threshold` | `number` |
| `very-conservative` | { `maxSpeechDurationS`: `number` = 20; `minSilenceDurationMs`: `number` = 300; `minSpeechDurationMs`: `number` = 750; `speechPadMs`: `number` = 10; `threshold`: `number` = 0.8 } |
| `very-conservative.maxSpeechDurationS` | `number` |
| `very-conservative.minSilenceDurationMs` | `number` |
| `very-conservative.minSpeechDurationMs` | `number` |
| `very-conservative.speechPadMs` | `number` |
| `very-conservative.threshold` | `number` |
| `very-sensitive` | { `maxSpeechDurationS`: `number` = 15; `minSilenceDurationMs`: `number` = 50; `minSpeechDurationMs`: `number` = 100; `speechPadMs`: `number` = 100; `threshold`: `number` = 0.2 } |
| `very-sensitive.maxSpeechDurationS` | `number` |
| `very-sensitive.minSilenceDurationMs` | `number` |
| `very-sensitive.minSpeechDurationMs` | `number` |
| `very-sensitive.speechPadMs` | `number` |
| `very-sensitive.threshold` | `number` |

#### Defined in

[realtime-transcription/types.ts:69](https://github.com/mybigday/whisper.rn/blob/3a63101/src/realtime-transcription/types.ts#L69)
