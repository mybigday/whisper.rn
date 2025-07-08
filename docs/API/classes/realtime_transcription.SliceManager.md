[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / SliceManager

# Class: SliceManager

[realtime-transcription](../modules/realtime_transcription.md).SliceManager

## Table of contents

### Constructors

- [constructor](realtime_transcription.SliceManager.md#constructor)

### Methods

- [addAudioData](realtime_transcription.SliceManager.md#addaudiodata)
- [forceNextSlice](realtime_transcription.SliceManager.md#forcenextslice)
- [getAudioDataForTranscription](realtime_transcription.SliceManager.md#getaudiodatafortranscription)
- [getCurrentSliceInfo](realtime_transcription.SliceManager.md#getcurrentsliceinfo)
- [getMemoryUsage](realtime_transcription.SliceManager.md#getmemoryusage)
- [getSliceByIndex](realtime_transcription.SliceManager.md#getslicebyindex)
- [getSliceForTranscription](realtime_transcription.SliceManager.md#getslicefortranscription)
- [markSliceAsProcessed](realtime_transcription.SliceManager.md#marksliceasprocessed)
- [moveToNextTranscribeSlice](realtime_transcription.SliceManager.md#movetonexttranscribeslice)
- [reset](realtime_transcription.SliceManager.md#reset)

## Constructors

### constructor

• **new SliceManager**(`sliceDurationSec?`, `maxSlicesInMemory?`, `sampleRate?`)

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `sliceDurationSec` | `number` | `30` |
| `maxSlicesInMemory` | `number` | `1` |
| `sampleRate` | `number` | `16000` |

#### Defined in

[realtime-transcription/SliceManager.ts:16](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L16)

## Methods

### addAudioData

▸ **addAudioData**(`audioData`): `Object`

Add audio data to the current slice

#### Parameters

| Name | Type |
| :------ | :------ |
| `audioData` | `Uint8Array` |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `slice?` | [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md) |

#### Defined in

[realtime-transcription/SliceManager.ts:29](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L29)

___

### forceNextSlice

▸ **forceNextSlice**(): `Object`

Force move to the next slice, finalizing the current one regardless of capacity

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `slice?` | [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md) |

#### Defined in

[realtime-transcription/SliceManager.ts:234](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L234)

___

### getAudioDataForTranscription

▸ **getAudioDataForTranscription**(`sliceIndex`): ``null`` \| `Uint8Array`

Get audio data for transcription (base64 encoded)

#### Parameters

| Name | Type |
| :------ | :------ |
| `sliceIndex` | `number` |

#### Returns

``null`` \| `Uint8Array`

#### Defined in

[realtime-transcription/SliceManager.ts:138](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L138)

___

### getCurrentSliceInfo

▸ **getCurrentSliceInfo**(): `Object`

Get current slice information

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `currentSliceIndex` | `number` |
| `memoryUsage` | [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md) |
| `totalSlices` | `number` |
| `transcribeSliceIndex` | `number` |

#### Defined in

[realtime-transcription/SliceManager.ts:222](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L222)

___

### getMemoryUsage

▸ **getMemoryUsage**(): [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md)

Get memory usage statistics

#### Returns

[`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md)

#### Defined in

[realtime-transcription/SliceManager.ts:185](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L185)

___

### getSliceByIndex

▸ **getSliceByIndex**(`sliceIndex`): ``null`` \| [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md)

Get a slice by index

#### Parameters

| Name | Type |
| :------ | :------ |
| `sliceIndex` | `number` |

#### Returns

``null`` \| [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md)

#### Defined in

[realtime-transcription/SliceManager.ts:151](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L151)

___

### getSliceForTranscription

▸ **getSliceForTranscription**(): ``null`` \| [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md)

Get a slice for transcription

#### Returns

``null`` \| [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md)

#### Defined in

[realtime-transcription/SliceManager.ts:106](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L106)

___

### markSliceAsProcessed

▸ **markSliceAsProcessed**(`sliceIndex`): `void`

Mark a slice as processed

#### Parameters

| Name | Type |
| :------ | :------ |
| `sliceIndex` | `number` |

#### Returns

`void`

#### Defined in

[realtime-transcription/SliceManager.ts:121](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L121)

___

### moveToNextTranscribeSlice

▸ **moveToNextTranscribeSlice**(): `void`

Move to the next slice for transcription

#### Returns

`void`

#### Defined in

[realtime-transcription/SliceManager.ts:131](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L131)

___

### reset

▸ **reset**(): `void`

Reset all slices and indices

#### Returns

`void`

#### Defined in

[realtime-transcription/SliceManager.ts:205](https://github.com/mybigday/whisper.rn/blob/4ad9647/src/realtime-transcription/SliceManager.ts#L205)
