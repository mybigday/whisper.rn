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

[index.ts:635](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L635)

## Properties

### contextId

• `Readonly` **contextId**: `number`

#### Defined in

[index.ts:623](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L623)

___

### hopSamples

• `Readonly` **hopSamples**: `number`

Samples covered by each probability (512 = 32 ms for Silero)

#### Defined in

[index.ts:631](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L631)

___

### id

• `Readonly` **id**: `number`

#### Defined in

[index.ts:625](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L625)

___

### sampleRate

• `Readonly` **sampleRate**: `number`

Sample rate the stream expects (16000)

#### Defined in

[index.ts:628](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L628)

## Methods

### flush

▸ **flush**(): `Promise`<`number`[]\>

End of audio: resolves with the probabilities still pending (a partial hop is zero padded).

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:666](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L666)

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

[index.ts:659](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L659)

___

### release

▸ **release**(): `Promise`<`void`\>

Free the stream. Releasing the context frees its streams as well.

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:678](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L678)

___

### reset

▸ **reset**(): `Promise`<`void`\>

Clear the model state and buffered audio to start a new audio stream.

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:672](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L672)
