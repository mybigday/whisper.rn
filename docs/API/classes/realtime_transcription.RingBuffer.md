[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RingBuffer

# Class: RingBuffer

[realtime-transcription](../modules/realtime_transcription.md).RingBuffer

RingBuffer - A fixed-size circular buffer for audio data

This class implements a ring buffer (circular buffer) that maintains
a fixed memory footprint regardless of how much data is written.
It's designed for pre-recording audio where we only need to keep
the last N seconds of audio before speech is detected.

Key features:
- Fixed memory allocation (no unbounded growth)
- O(1) write operations
- Preserves most recent data when buffer is full

## Table of contents

### Constructors

- [constructor](realtime_transcription.RingBuffer.md#constructor)

### Methods

- [clear](realtime_transcription.RingBuffer.md#clear)
- [getCapacity](realtime_transcription.RingBuffer.md#getcapacity)
- [getFillRatio](realtime_transcription.RingBuffer.md#getfillratio)
- [getLength](realtime_transcription.RingBuffer.md#getlength)
- [isEmpty](realtime_transcription.RingBuffer.md#isempty)
- [isFull](realtime_transcription.RingBuffer.md#isfull)
- [read](realtime_transcription.RingBuffer.md#read)
- [write](realtime_transcription.RingBuffer.md#write)

## Constructors

### constructor

• **new RingBuffer**(`maxBytes`)

Create a new RingBuffer

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `maxBytes` | `number` | Maximum buffer size in bytes |

#### Defined in

[realtime-transcription/RingBuffer.ts:27](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L27)

## Methods

### clear

▸ **clear**(): `void`

Clear the buffer and reset indices

#### Returns

`void`

#### Defined in

[realtime-transcription/RingBuffer.ts:102](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L102)

___

### getCapacity

▸ **getCapacity**(): `number`

Get the maximum capacity of the buffer

#### Returns

`number`

Maximum buffer size in bytes

#### Defined in

[realtime-transcription/RingBuffer.ts:120](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L120)

___

### getFillRatio

▸ **getFillRatio**(): `number`

Get the fill percentage of the buffer

#### Returns

`number`

Number between 0 and 1 representing how full the buffer is

#### Defined in

[realtime-transcription/RingBuffer.ts:144](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L144)

___

### getLength

▸ **getLength**(): `number`

Get the current amount of data in the buffer

#### Returns

`number`

Number of bytes currently in the buffer

#### Defined in

[realtime-transcription/RingBuffer.ts:112](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L112)

___

### isEmpty

▸ **isEmpty**(): `boolean`

Check if the buffer is empty

#### Returns

`boolean`

true if buffer contains no data

#### Defined in

[realtime-transcription/RingBuffer.ts:128](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L128)

___

### isFull

▸ **isFull**(): `boolean`

Check if the buffer is full

#### Returns

`boolean`

true if buffer has reached capacity

#### Defined in

[realtime-transcription/RingBuffer.ts:136](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L136)

___

### read

▸ **read**(): `Uint8Array`

Read all available data from the buffer in correct order
Does NOT clear the buffer (use clear() for that)

#### Returns

`Uint8Array`

Uint8Array containing the buffered data in order

#### Defined in

[realtime-transcription/RingBuffer.ts:75](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L75)

___

### write

▸ **write**(`data`): `void`

Write audio data to the buffer
If data exceeds buffer capacity, oldest data is overwritten

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `data` | `Uint8Array` | Audio data to write |

#### Returns

`void`

#### Defined in

[realtime-transcription/RingBuffer.ts:37](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/RingBuffer.ts#L37)
