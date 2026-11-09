[whisper.rn](../README.md) / realtime-transcription

# Module: realtime-transcription

## Table of contents

### Classes

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
- [RealtimeVadEvent](../interfaces/realtime_transcription.RealtimeVadEvent.md)
- [WavFileWriterFs](../interfaces/realtime_transcription.WavFileWriterFs.md)

### Type Aliases

- [RealtimeTranscriberDependencies](realtime_transcription.md#realtimetranscriberdependencies)
- [StreamVadContextLike](realtime_transcription.md#streamvadcontextlike)
- [StreamVadOptions](realtime_transcription.md#streamvadoptions)
- [VadSegmenterEvent](realtime_transcription.md#vadsegmenterevent)
- [VadStreamLike](realtime_transcription.md#vadstreamlike)

### Variables

- [VAD\_PRESETS](realtime_transcription.md#vad_presets)

## Type Aliases

### RealtimeTranscriberDependencies

Ƭ **RealtimeTranscriberDependencies**: { `audioStream`: [`AudioStreamInterface`](../interfaces/realtime_transcription.AudioStreamInterface.md) ; `fs?`: [`WavFileWriterFs`](../interfaces/realtime_transcription.WavFileWriterFs.md) ; `vadContext?`: `RealtimeVadContextLike`  } & { `parakeetContext?`: `never` ; `whisperContext`: `WhisperContextLike`  } \| { `parakeetContext`: `ParakeetContextLike` ; `whisperContext?`: `never`  }

Dependencies for realtime transcription. Provide exactly one transcription
context: `whisperContext` or `parakeetContext`.

#### Defined in

[realtime-transcription/types.ts:328](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L328)

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

[realtime-transcription/StreamVad.ts:15](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L15)

___

### StreamVadOptions

Ƭ **StreamVadOptions**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `logger?` | (`message`: `string`) => `void` |
| `vadOptions?` | [`VadOptions`](index.md#vadoptions) |
| `vadPreset?` | keyof typeof [`VAD_PRESETS`](realtime_transcription.md#vad_presets) |

#### Defined in

[realtime-transcription/StreamVad.ts:21](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L21)

___

### VadSegmenterEvent

Ƭ **VadSegmenterEvent**: { `prob`: `number` ; `sample`: `number` ; `type`: ``"start"``  } \| { `prob`: `number` ; `sample`: `number` ; `type`: ``"end"``  }

#### Defined in

[realtime-transcription/StreamingVadSegmenter.ts:3](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamingVadSegmenter.ts#L3)

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

[realtime-transcription/StreamVad.ts:7](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/StreamVad.ts#L7)

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

[realtime-transcription/types.ts:66](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/realtime-transcription/types.ts#L66)
