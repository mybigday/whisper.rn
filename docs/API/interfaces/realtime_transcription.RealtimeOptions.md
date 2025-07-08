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
- [debug](realtime_transcription.RealtimeOptions.md#debug)
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

[realtime-transcription/types.ts:156](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L156)

___

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:175](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L175)

___

### audioSliceSec

• `Optional` **audioSliceSec**: `number`

#### Defined in

[realtime-transcription/types.ts:155](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L155)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:178](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L178)

___

### autoSliceOnSpeechEnd

• `Optional` **autoSliceOnSpeechEnd**: `boolean`

#### Defined in

[realtime-transcription/types.ts:164](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L164)

___

### autoSliceThreshold

• `Optional` **autoSliceThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:165](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L165)

___

### debug

• `Optional` **debug**: `boolean`

#### Defined in

[realtime-transcription/types.ts:181](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L181)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

#### Defined in

[realtime-transcription/types.ts:171](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L171)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

#### Defined in

[realtime-transcription/types.ts:157](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L157)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

#### Defined in

[realtime-transcription/types.ts:172](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L172)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeOptions`](../modules/index.md#transcribeoptions)

#### Defined in

[realtime-transcription/types.ts:168](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L168)

___

### vadOptions

• `Optional` **vadOptions**: [`VadOptions`](../modules/index.md#vadoptions)

#### Defined in

[realtime-transcription/types.ts:160](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L160)

___

### vadPreset

• `Optional` **vadPreset**: ``"continuous"`` \| ``"default"`` \| ``"sensitive"`` \| ``"very-sensitive"`` \| ``"conservative"`` \| ``"very-conservative"`` \| ``"meeting"`` \| ``"noisy"``

#### Defined in

[realtime-transcription/types.ts:161](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L161)
