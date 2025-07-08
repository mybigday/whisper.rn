[whisper.rn](../README.md) / realtime-transcription

# Module: realtime-transcription

## Table of contents

### Classes

- [RealtimeTranscriber](../classes/realtime_transcription.RealtimeTranscriber.md)
- [SliceManager](../classes/realtime_transcription.SliceManager.md)

### Interfaces

- [AudioSlice](../interfaces/realtime_transcription.AudioSlice.md)
- [AudioSliceNoData](../interfaces/realtime_transcription.AudioSliceNoData.md)
- [AudioStreamConfig](../interfaces/realtime_transcription.AudioStreamConfig.md)
- [AudioStreamData](../interfaces/realtime_transcription.AudioStreamData.md)
- [AudioStreamInterface](../interfaces/realtime_transcription.AudioStreamInterface.md)
- [MemoryUsage](../interfaces/realtime_transcription.MemoryUsage.md)
- [RealtimeOptions](../interfaces/realtime_transcription.RealtimeOptions.md)
- [RealtimeTranscriberCallbacks](../interfaces/realtime_transcription.RealtimeTranscriberCallbacks.md)
- [RealtimeTranscriberContexts](../interfaces/realtime_transcription.RealtimeTranscriberContexts.md)
- [RealtimeTranscriberDependencies](../interfaces/realtime_transcription.RealtimeTranscriberDependencies.md)
- [StatsEvent](../interfaces/realtime_transcription.StatsEvent.md)
- [TranscribeEvent](../interfaces/realtime_transcription.TranscribeEvent.md)
- [VadEvent](../interfaces/realtime_transcription.VadEvent.md)
- [WavFileWriterFs](../interfaces/realtime_transcription.WavFileWriterFs.md)

### Variables

- [VAD\_PRESETS](realtime_transcription.md#vad_presets)

## Variables

### VAD\_PRESETS

• `Const` **VAD\_PRESETS**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `CONSERVATIVE` | { `maxSpeechDurationS`: `number` = 25; `minSilenceDurationMs`: `number` = 200; `minSpeechDurationMs`: `number` = 500; `samplesOverlap`: `number` = 0.05; `speechPadMs`: `number` = 20; `threshold`: `number` = 0.7 } |
| `CONSERVATIVE.maxSpeechDurationS` | `number` |
| `CONSERVATIVE.minSilenceDurationMs` | `number` |
| `CONSERVATIVE.minSpeechDurationMs` | `number` |
| `CONSERVATIVE.samplesOverlap` | `number` |
| `CONSERVATIVE.speechPadMs` | `number` |
| `CONSERVATIVE.threshold` | `number` |
| `CONTINUOUS_SPEECH` | { `maxSpeechDurationS`: `number` = 60; `minSilenceDurationMs`: `number` = 300; `minSpeechDurationMs`: `number` = 200; `samplesOverlap`: `number` = 0.15; `speechPadMs`: `number` = 50; `threshold`: `number` = 0.4 } |
| `CONTINUOUS_SPEECH.maxSpeechDurationS` | `number` |
| `CONTINUOUS_SPEECH.minSilenceDurationMs` | `number` |
| `CONTINUOUS_SPEECH.minSpeechDurationMs` | `number` |
| `CONTINUOUS_SPEECH.samplesOverlap` | `number` |
| `CONTINUOUS_SPEECH.speechPadMs` | `number` |
| `CONTINUOUS_SPEECH.threshold` | `number` |
| `DEFAULT` | { `maxSpeechDurationS`: `number` = 30; `minSilenceDurationMs`: `number` = 100; `minSpeechDurationMs`: `number` = 250; `samplesOverlap`: `number` = 0.1; `speechPadMs`: `number` = 30; `threshold`: `number` = 0.5 } |
| `DEFAULT.maxSpeechDurationS` | `number` |
| `DEFAULT.minSilenceDurationMs` | `number` |
| `DEFAULT.minSpeechDurationMs` | `number` |
| `DEFAULT.samplesOverlap` | `number` |
| `DEFAULT.speechPadMs` | `number` |
| `DEFAULT.threshold` | `number` |
| `MEETING` | { `maxSpeechDurationS`: `number` = 45; `minSilenceDurationMs`: `number` = 150; `minSpeechDurationMs`: `number` = 300; `samplesOverlap`: `number` = 0.2; `speechPadMs`: `number` = 75; `threshold`: `number` = 0.45 } |
| `MEETING.maxSpeechDurationS` | `number` |
| `MEETING.minSilenceDurationMs` | `number` |
| `MEETING.minSpeechDurationMs` | `number` |
| `MEETING.samplesOverlap` | `number` |
| `MEETING.speechPadMs` | `number` |
| `MEETING.threshold` | `number` |
| `NOISY_ENVIRONMENT` | { `maxSpeechDurationS`: `number` = 25; `minSilenceDurationMs`: `number` = 100; `minSpeechDurationMs`: `number` = 400; `samplesOverlap`: `number` = 0.1; `speechPadMs`: `number` = 40; `threshold`: `number` = 0.75 } |
| `NOISY_ENVIRONMENT.maxSpeechDurationS` | `number` |
| `NOISY_ENVIRONMENT.minSilenceDurationMs` | `number` |
| `NOISY_ENVIRONMENT.minSpeechDurationMs` | `number` |
| `NOISY_ENVIRONMENT.samplesOverlap` | `number` |
| `NOISY_ENVIRONMENT.speechPadMs` | `number` |
| `NOISY_ENVIRONMENT.threshold` | `number` |
| `SENSITIVE` | { `maxSpeechDurationS`: `number` = 15; `minSilenceDurationMs`: `number` = 50; `minSpeechDurationMs`: `number` = 100; `samplesOverlap`: `number` = 0.2; `speechPadMs`: `number` = 50; `threshold`: `number` = 0.3 } |
| `SENSITIVE.maxSpeechDurationS` | `number` |
| `SENSITIVE.minSilenceDurationMs` | `number` |
| `SENSITIVE.minSpeechDurationMs` | `number` |
| `SENSITIVE.samplesOverlap` | `number` |
| `SENSITIVE.speechPadMs` | `number` |
| `SENSITIVE.threshold` | `number` |
| `VERY_CONSERVATIVE` | { `maxSpeechDurationS`: `number` = 20; `minSilenceDurationMs`: `number` = 300; `minSpeechDurationMs`: `number` = 750; `samplesOverlap`: `number` = 0.05; `speechPadMs`: `number` = 10; `threshold`: `number` = 0.8 } |
| `VERY_CONSERVATIVE.maxSpeechDurationS` | `number` |
| `VERY_CONSERVATIVE.minSilenceDurationMs` | `number` |
| `VERY_CONSERVATIVE.minSpeechDurationMs` | `number` |
| `VERY_CONSERVATIVE.samplesOverlap` | `number` |
| `VERY_CONSERVATIVE.speechPadMs` | `number` |
| `VERY_CONSERVATIVE.threshold` | `number` |
| `VERY_SENSITIVE` | { `maxSpeechDurationS`: `number` = 15; `minSilenceDurationMs`: `number` = 50; `minSpeechDurationMs`: `number` = 100; `samplesOverlap`: `number` = 0.3; `speechPadMs`: `number` = 100; `threshold`: `number` = 0.2 } |
| `VERY_SENSITIVE.maxSpeechDurationS` | `number` |
| `VERY_SENSITIVE.minSilenceDurationMs` | `number` |
| `VERY_SENSITIVE.minSpeechDurationMs` | `number` |
| `VERY_SENSITIVE.samplesOverlap` | `number` |
| `VERY_SENSITIVE.speechPadMs` | `number` |
| `VERY_SENSITIVE.threshold` | `number` |

#### Defined in

[realtime-transcription/types.ts:41](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L41)
