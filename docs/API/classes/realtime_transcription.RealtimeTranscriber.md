[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeTranscriber

# Class: RealtimeTranscriber

[realtime-transcription](../modules/realtime_transcription.md).RealtimeTranscriber

RealtimeTranscriber provides real-time audio transcription with VAD support.

Features:
- Automatic slice management based on duration
- VAD-based speech detection and auto-slicing
- Configurable auto-slice mechanism that triggers on speech_end/silence events
- Memory management for audio slices
- Queue-based transcription processing

## Table of contents

### Constructors

- [constructor](realtime_transcription.RealtimeTranscriber.md#constructor)

### Methods

- [getStatistics](realtime_transcription.RealtimeTranscriber.md#getstatistics)
- [getTranscriptionResults](realtime_transcription.RealtimeTranscriber.md#gettranscriptionresults)
- [nextSlice](realtime_transcription.RealtimeTranscriber.md#nextslice)
- [release](realtime_transcription.RealtimeTranscriber.md#release)
- [reset](realtime_transcription.RealtimeTranscriber.md#reset)
- [start](realtime_transcription.RealtimeTranscriber.md#start)
- [stop](realtime_transcription.RealtimeTranscriber.md#stop)
- [updateCallbacks](realtime_transcription.RealtimeTranscriber.md#updatecallbacks)
- [updateVadOptions](realtime_transcription.RealtimeTranscriber.md#updatevadoptions)

## Constructors

### constructor

• **new RealtimeTranscriber**(`dependencies`, `options?`, `callbacks?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `dependencies` | [`RealtimeTranscriberDependencies`](../modules/realtime_transcription.md#realtimetranscriberdependencies) |
| `options` | [`RealtimeOptions`](../interfaces/realtime_transcription.RealtimeOptions.md) |
| `callbacks` | [`RealtimeTranscriberCallbacks`](../interfaces/realtime_transcription.RealtimeTranscriberCallbacks.md) |

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:103](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L103)

## Methods

### getStatistics

▸ **getStatistics**(): `Object`

Get current statistics

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `audioStats` | { `accumulatedSamples`: `number` ; `isRecording`: `boolean`  } |
| `audioStats.accumulatedSamples` | `number` |
| `audioStats.isRecording` | `boolean` |
| `isActive` | `boolean` |
| `isTranscribing` | `boolean` |
| `sliceStats` | { `currentSliceIndex`: `number` ; `memoryUsage`: [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md) ; `totalSlices`: `number` ; `transcribeSliceIndex`: `number`  } |
| `sliceStats.currentSliceIndex` | `number` |
| `sliceStats.memoryUsage` | [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md) |
| `sliceStats.totalSlices` | `number` |
| `sliceStats.transcribeSliceIndex` | `number` |
| `vadEnabled` | `boolean` |
| `vadStats` | ``null`` \| { `contextAvailable`: `boolean` = !!this.vadContext; `enabled`: `boolean` = true; `lastSpeechDetectedTime`: `number`  } |

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:664](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L664)

___

### getTranscriptionResults

▸ **getTranscriptionResults**(): { `slice`: [`AudioSliceNoData`](../interfaces/realtime_transcription.AudioSliceNoData.md) ; `transcribeEvent`: [`RealtimeTranscribeEvent`](../interfaces/realtime_transcription.RealtimeTranscribeEvent.md)  }[]

Get all transcription results

#### Returns

{ `slice`: [`AudioSliceNoData`](../interfaces/realtime_transcription.AudioSliceNoData.md) ; `transcribeEvent`: [`RealtimeTranscribeEvent`](../interfaces/realtime_transcription.RealtimeTranscribeEvent.md)  }[]

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:687](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L687)

___

### nextSlice

▸ **nextSlice**(): `Promise`<`void`\>

Force move to the next slice, finalizing the current one regardless of capacity

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:697](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L697)

___

### release

▸ **release**(): `Promise`<`void`\>

Release all resources

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:790](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L790)

___

### reset

▸ **reset**(): `void`

Reset all components

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:752](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L752)

___

### start

▸ **start**(): `Promise`<`void`\>

Start realtime transcription

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:164](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L164)

___

### stop

▸ **stop**(): `Promise`<`void`\>

Stop realtime transcription

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:214](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L214)

___

### updateCallbacks

▸ **updateCallbacks**(`callbacks`): `void`

Update callbacks

#### Parameters

| Name | Type |
| :------ | :------ |
| `callbacks` | `Partial`<[`RealtimeTranscriberCallbacks`](../interfaces/realtime_transcription.RealtimeTranscriberCallbacks.md)\> |

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:648](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L648)

___

### updateVadOptions

▸ **updateVadOptions**(`options`): `void`

Update VAD options dynamically (delegates to VAD context)

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Partial`<[`VadOptions`](../modules/index.md#vadoptions)\> |

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:655](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/realtime-transcription/RealtimeTranscriber.ts#L655)
