[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / SliceManager

# Class: SliceManager

[realtime-transcription](../modules/realtime_transcription.md).SliceManager

Audio slices of a realtime session: one per speech segment, closed by
`forceNextSlice()`. Slices grow as audio arrives (their length is bounded by
the segmenter's maximum duration); only the newest `maxSlicesInMemory` keep
their audio.

Note: `sampleCount` counts bytes of PCM16, not samples.

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
- [truncateCurrentSlice](realtime_transcription.SliceManager.md#truncatecurrentslice)

## Constructors

### constructor

• **new SliceManager**(`maxSlicesInMemory?`)

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `maxSlicesInMemory` | `number` | `3` |

#### Defined in

[realtime-transcription/SliceManager.ts:47](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L47)

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
| `slice` | [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md) |

#### Defined in

[realtime-transcription/SliceManager.ts:54](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L54)

___

### forceNextSlice

▸ **forceNextSlice**(): `Object`

Close the current slice and move to the next one

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `slice?` | [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md) |

#### Defined in

[realtime-transcription/SliceManager.ts:235](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L235)

___

### getAudioDataForTranscription

▸ **getAudioDataForTranscription**(`sliceIndex`): ``null`` \| `Uint8Array`

The audio of a slice (a view of exactly its bytes)

#### Parameters

| Name | Type |
| :------ | :------ |
| `sliceIndex` | `number` |

#### Returns

``null`` \| `Uint8Array`

#### Defined in

[realtime-transcription/SliceManager.ts:129](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L129)

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

[realtime-transcription/SliceManager.ts:212](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L212)

___

### getMemoryUsage

▸ **getMemoryUsage**(): [`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md)

Get memory usage statistics

#### Returns

[`MemoryUsage`](../interfaces/realtime_transcription.MemoryUsage.md)

#### Defined in

[realtime-transcription/SliceManager.ts:175](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L175)

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

[realtime-transcription/SliceManager.ts:141](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L141)

___

### getSliceForTranscription

▸ **getSliceForTranscription**(): ``null`` \| [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md)

Get a slice for transcription

#### Returns

``null`` \| [`AudioSlice`](../interfaces/realtime_transcription.AudioSlice.md)

#### Defined in

[realtime-transcription/SliceManager.ts:97](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L97)

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

[realtime-transcription/SliceManager.ts:112](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L112)

___

### moveToNextTranscribeSlice

▸ **moveToNextTranscribeSlice**(): `void`

Move to the next slice for transcription

#### Returns

`void`

#### Defined in

[realtime-transcription/SliceManager.ts:122](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L122)

___

### reset

▸ **reset**(): `void`

Reset all slices and indices

#### Returns

`void`

#### Defined in

[realtime-transcription/SliceManager.ts:195](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L195)

___

### truncateCurrentSlice

▸ **truncateCurrentSlice**(`byteLength`): `void`

Shorten the current slice to its first `byteLength` bytes (no-op if it is
not longer), e.g. when its speech segment turns out to end earlier.

#### Parameters

| Name | Type |
| :------ | :------ |
| `byteLength` | `number` |

#### Returns

`void`

#### Defined in

[realtime-transcription/SliceManager.ts:225](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/SliceManager.ts#L225)
