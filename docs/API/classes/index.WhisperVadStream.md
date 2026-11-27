[whisper.rn](../README.md) / [index](../modules/index.md) / WhisperVadStream

# Class: WhisperVadStream

[index](../modules/index.md).WhisperVadStream

Stateful voice activity detection over live audio, created with
`WhisperVadContext.createStream()`.

The model state carries over between pushes, so every sample is analyzed
once and each probability takes all earlier audio into account (Silero's
exact streaming semantics). Calls on one stream run in call order.

## Table of contents

### Constructors

- [constructor](index.WhisperVadStream.md#constructor)

### Properties

- [contextId](index.WhisperVadStream.md#contextid)
- [hopSamples](index.WhisperVadStream.md#hopsamples)
- [id](index.WhisperVadStream.md#id)
- [sampleRate](index.WhisperVadStream.md#samplerate)

### Methods

- [flush](index.WhisperVadStream.md#flush)
- [push](index.WhisperVadStream.md#push)
- [release](index.WhisperVadStream.md#release)
- [reset](index.WhisperVadStream.md#reset)

## Constructors

### constructor

• **new WhisperVadStream**(`contextId`, `streamId`, `sampleRate`, `hopSamples`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `contextId` | `number` |
| `streamId` | `number` |
| `sampleRate` | `number` |
| `hopSamples` | `number` |

#### Defined in

[index.ts:672](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L672)

## Properties

### contextId

• `Readonly` **contextId**: `number`

#### Defined in

[index.ts:660](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L660)

___

### hopSamples

• `Readonly` **hopSamples**: `number`

Samples covered by each probability (512 = 32 ms for Silero)

#### Defined in

[index.ts:668](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L668)

___

### id

• `Readonly` **id**: `number`

#### Defined in

[index.ts:662](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L662)

___

### sampleRate

• `Readonly` **sampleRate**: `number`

Sample rate the stream expects (16000)

#### Defined in

[index.ts:665](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L665)

## Methods

### flush

▸ **flush**(): `Promise`<`number`[]\>

End of audio: resolves with the probabilities still pending (a partial hop is zero padded).

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:703](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L703)

___

### push

▸ **push**(`audio`): `Promise`<`number`[]\>

Feed 16 kHz mono PCM16 audio of any length. Resolves with the speech
probabilities that became available, in order: probability `i` of the
stream covers samples `[i * hopSamples, (i + 1) * hopSamples)`.

#### Parameters

| Name | Type |
| :------ | :------ |
| `audio` | `ArrayBuffer` \| `Uint8Array` |

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:696](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L696)

___

### release

▸ **release**(): `Promise`<`void`\>

Free the stream. Releasing the context frees its streams as well.

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:715](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L715)

___

### reset

▸ **reset**(): `Promise`<`void`\>

Clear the model state and buffered audio to start a new audio stream.

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:709](https://github.com/mybigday/whisper.rn/blob/90f6961/src/index.ts#L709)
