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
- [logger](realtime_transcription.RealtimeOptions.md#logger)
- [maxSlicesInMemory](realtime_transcription.RealtimeOptions.md#maxslicesinmemory)
- [promptPreviousSlices](realtime_transcription.RealtimeOptions.md#promptpreviousslices)
- [transcribeOptions](realtime_transcription.RealtimeOptions.md#transcribeoptions)
- [vadOptions](realtime_transcription.RealtimeOptions.md#vadoptions)
- [vadPreset](realtime_transcription.RealtimeOptions.md#vadpreset)

## Properties

### audioMinSec

• `Optional` **audioMinSec**: `number`

#### Defined in

[realtime-transcription/types.ts:182](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L182)

___

### audioOutputPath

• `Optional` **audioOutputPath**: `string`

#### Defined in

[realtime-transcription/types.ts:201](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L201)

___

### audioSliceSec

• `Optional` **audioSliceSec**: `number`

#### Defined in

[realtime-transcription/types.ts:181](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L181)

___

### audioStreamConfig

• `Optional` **audioStreamConfig**: [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md)

#### Defined in

[realtime-transcription/types.ts:204](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L204)

___

### autoSliceOnSpeechEnd

• `Optional` **autoSliceOnSpeechEnd**: `boolean`

#### Defined in

[realtime-transcription/types.ts:190](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L190)

___

### autoSliceThreshold

• `Optional` **autoSliceThreshold**: `number`

#### Defined in

[realtime-transcription/types.ts:191](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L191)

___

### initialPrompt

• `Optional` **initialPrompt**: `string`

#### Defined in

[realtime-transcription/types.ts:197](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L197)

___

### logger

• `Optional` **logger**: (`message`: `string`) => `void`

#### Type declaration

▸ (`message`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `message` | `string` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:207](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L207)

___

### maxSlicesInMemory

• `Optional` **maxSlicesInMemory**: `number`

#### Defined in

[realtime-transcription/types.ts:183](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L183)

___

### promptPreviousSlices

• `Optional` **promptPreviousSlices**: `boolean`

#### Defined in

[realtime-transcription/types.ts:198](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L198)

___

### transcribeOptions

• `Optional` **transcribeOptions**: [`TranscribeOptions`](../modules/index.md#transcribeoptions)

#### Defined in

[realtime-transcription/types.ts:194](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L194)

___

### vadOptions

• `Optional` **vadOptions**: [`VadOptions`](../modules/index.md#vadoptions)

#### Defined in

[realtime-transcription/types.ts:186](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L186)

___

### vadPreset

• `Optional` **vadPreset**: ``"continuous"`` \| ``"default"`` \| ``"sensitive"`` \| ``"very-sensitive"`` \| ``"conservative"`` \| ``"very-conservative"`` \| ``"meeting"`` \| ``"noisy"``

#### Defined in

[realtime-transcription/types.ts:187](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L187)
