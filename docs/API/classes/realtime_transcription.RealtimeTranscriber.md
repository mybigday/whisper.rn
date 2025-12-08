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
- [updateAutoSliceOptions](realtime_transcription.RealtimeTranscriber.md#updateautosliceoptions)
- [updateCallbacks](realtime_transcription.RealtimeTranscriber.md#updatecallbacks)
- [updateVadOptions](realtime_transcription.RealtimeTranscriber.md#updatevadoptions)
- [updateVadThrottleOptions](realtime_transcription.RealtimeTranscriber.md#updatevadthrottleoptions)

## Constructors

### constructor

• **new RealtimeTranscriber**(`dependencies`, `options?`, `callbacks?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `dependencies` | [`RealtimeTranscriberDependencies`](../interfaces/realtime_transcription.RealtimeTranscriberDependencies.md) |
| `options` | [`RealtimeOptions`](../interfaces/realtime_transcription.RealtimeOptions.md) |
| `callbacks` | [`RealtimeTranscriberCallbacks`](../interfaces/realtime_transcription.RealtimeTranscriberCallbacks.md) |

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:107](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L107)

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
| `autoSliceConfig` | { `enabled`: `boolean` ; `minDuration`: `number` ; `targetDuration`: `number` ; `threshold`: `number`  } |
| `autoSliceConfig.enabled` | `boolean` |
| `autoSliceConfig.minDuration` | `number` |
| `autoSliceConfig.targetDuration` | `number` |
| `autoSliceConfig.threshold` | `number` |
| `isActive` | `boolean` |
| `isTranscribing` | `boolean` |
| `sliceStats` | { `currentSliceIndex`: `number` ; `memoryUsage`: [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md) ; `totalSlices`: `number` ; `transcribeSliceIndex`: `number`  } |
| `sliceStats.currentSliceIndex` | `number` |
| `sliceStats.memoryUsage` | [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md) |
| `sliceStats.totalSlices` | `number` |
| `sliceStats.transcribeSliceIndex` | `number` |
| `vadEnabled` | `boolean` |
| `vadStats` | ``null`` \| { `contextAvailable`: `boolean` = !!this.vadContext; `enabled`: `boolean` = true; `isProcessing`: `boolean` ; `lastSpeechDetectedTime`: `number` ; `queueSize`: `number` ; `skipRatio`: `number` ; `skippedCount`: `number` ; `throttleMs`: `number`  } |

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:979](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L979)

___

### getTranscriptionResults

▸ **getTranscriptionResults**(): { `slice`: [`AudioSliceNoData`](../interfaces/realtime_transcription.AudioSliceNoData.md) ; `transcribeEvent`: [`RealtimeTranscribeEvent`](../interfaces/realtime_transcription.RealtimeTranscribeEvent.md)  }[]

Get all transcription results

#### Returns

{ `slice`: [`AudioSliceNoData`](../interfaces/realtime_transcription.AudioSliceNoData.md) ; `transcribeEvent`: [`RealtimeTranscribeEvent`](../interfaces/realtime_transcription.RealtimeTranscribeEvent.md)  }[]

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1013](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L1013)

___

### nextSlice

▸ **nextSlice**(): `Promise`<`void`\>

Force move to the next slice, finalizing the current one regardless of capacity

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1023](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L1023)

___

### release

▸ **release**(): `Promise`<`void`\>

Release all resources

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1117](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L1117)

___

### reset

▸ **reset**(): `void`

Reset all components

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:1080](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L1080)

___

### start

▸ **start**(): `Promise`<`void`\>

Start realtime transcription

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:163](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L163)

___

### stop

▸ **stop**(): `Promise`<`void`\>

Stop realtime transcription

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:213](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L213)

___

### updateAutoSliceOptions

▸ **updateAutoSliceOptions**(`options`): `void`

Update auto-slice options dynamically

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Object` |
| `options.autoSliceOnSpeechEnd?` | `boolean` |
| `options.autoSliceThreshold?` | `number` |

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:943](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L943)

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

[realtime-transcription/RealtimeTranscriber.ts:929](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L929)

___

### updateVadOptions

▸ **updateVadOptions**(`options`): `void`

Update VAD options dynamically

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Partial`<[`VadOptions`](../modules/index.md#vadoptions)\> |

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:936](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L936)

___

### updateVadThrottleOptions

▸ **updateVadThrottleOptions**(`options`): `void`

Update VAD throttling options dynamically for low-end CPU optimization

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Object` |
| `options.vadSkipRatio?` | `number` |
| `options.vadThrottleMs?` | `number` |

#### Returns

`void`

#### Defined in

[realtime-transcription/RealtimeTranscriber.ts:961](https://github.com/mybigday/whisper.rn/blob/42bb2d5/src/realtime-transcription/RealtimeTranscriber.ts#L961)
