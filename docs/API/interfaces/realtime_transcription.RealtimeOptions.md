[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeOptions

# Interface: RealtimeOptions

[realtime-transcription](../modules/realtime_transcription.md).RealtimeOptions

## Table of contents

### Properties

- [audioMinSec](realtime_transcription.RealtimeOptions.md#audiominsec)
- [audioOutputPath](realtime_transcription.RealtimeOptions.md#audiooutputpath)
- [audioSliceSec](realtime_transcription.RealtimeOptions.md#audioslicesec)
- [audioStreamConfig](realtime_transcription.RealtimeOptions.md#audiostreamconfig)
- [autoSliceOnSpeechEnd](realtime_transcription.RealtimeOptions.md#autosliceonspeechend)
- [autoSliceThreshold](realtime_transcription.RealtimeOptions.md#autoslicethreshold)
- [initialPrompt](realtime_transcription.RealtimeOptions.md#initialprompt)
- [maxSlicesInMemory](realtime_transcription.RealtimeOptions.md#maxslicesinmemory)
- [promptPreviousSlices](realtime_transcription.RealtimeOptions.md#promptpreviousslices)
- [transcribeOptions](realtime_transcription.RealtimeOptions.md#transcribeoptions)
- [vadOptions](realtime_transcription.RealtimeOptions.md#vadoptions)
- [vadPreset](realtime_transcription.RealtimeOptions.md#vadpreset)

## Properties

### audioMinSec

• `Optional` **audioMinSec**: `number`

#### Defined in

[realtime-transcription/types.ts:162](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L162)

___

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:181](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L181)

___

### audioSliceSec

• `Optional` **audioSliceSec**: `number`

#### Defined in

[realtime-transcription/types.ts:161](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L161)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:184](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L184)

___

### autoSliceOnSpeechEnd

• `Optional` **autoSliceOnSpeechEnd**: `boolean`

#### Defined in

[realtime-transcription/types.ts:170](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L170)

___

### autoSliceThreshold

• `Optional` **autoSliceThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:171](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L171)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

#### Defined in

[realtime-transcription/types.ts:177](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L177)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

#### Defined in

[realtime-transcription/types.ts:163](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L163)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

#### Defined in

[realtime-transcription/types.ts:178](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L178)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeFileOptions`](../modules/index.md#transcribefileoptions)

#### Defined in

[realtime-transcription/types.ts:174](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L174)

___

### vadOptions

• `Optional` **vadOptions**: [`VadOptions`](../modules/index.md#vadoptions)

#### Defined in

[realtime-transcription/types.ts:166](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L166)

___

### vadPreset

• `Optional` **vadPreset**: ``"DEFAULT"`` \| ``"SENSITIVE"`` \| ``"VERY_SENSITIVE"`` \| ``"CONSERVATIVE"`` \| ``"VERY_CONSERVATIVE"`` \| ``"CONTINUOUS_SPEECH"`` \| ``"MEETING"`` \| ``"NOISY_ENVIRONMENT"``

#### Defined in

[realtime-transcription/types.ts:167](https://github.com/mybigday/whisper.rn/blob/874c510/src/realtime-transcription/types.ts#L167)
