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

[realtime-transcription/types.ts:160](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L160)

___

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:179](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L179)

___

### audioSliceSec

• `Optional` **audioSliceSec**: `number`

#### Defined in

[realtime-transcription/types.ts:159](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L159)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:182](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L182)

___

### autoSliceOnSpeechEnd

• `Optional` **autoSliceOnSpeechEnd**: `boolean`

#### Defined in

[realtime-transcription/types.ts:168](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L168)

___

### autoSliceThreshold

• `Optional` **autoSliceThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:169](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L169)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

#### Defined in

[realtime-transcription/types.ts:175](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L175)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

#### Defined in

[realtime-transcription/types.ts:161](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L161)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

#### Defined in

[realtime-transcription/types.ts:176](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L176)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeOptions`](../modules/index.md#transcribeoptions)

#### Defined in

[realtime-transcription/types.ts:172](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L172)

___

### vadOptions

• `Optional` **vadOptions**: [`VadOptions`](../modules/index.md#vadoptions)

#### Defined in

[realtime-transcription/types.ts:164](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L164)

___

### vadPreset

• `Optional` **vadPreset**: ``"DEFAULT"`` \| ``"SENSITIVE"`` \| ``"VERY_SENSITIVE"`` \| ``"CONSERVATIVE"`` \| ``"VERY_CONSERVATIVE"`` \| ``"CONTINUOUS_SPEECH"`` \| ``"MEETING"`` \| ``"NOISY_ENVIRONMENT"``

#### Defined in

[realtime-transcription/types.ts:165](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/types.ts#L165)
