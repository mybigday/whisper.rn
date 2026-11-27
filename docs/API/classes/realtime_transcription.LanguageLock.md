[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / LanguageLock

# Class: LanguageLock

[realtime-transcription](../modules/realtime_transcription.md).LanguageLock

Keeps an auto-detected source language stable across slices: per-slice
detection on short audio flips between languages, and every flip changes
the transcript. Once a language was detected confidently `lockAfter` times
it is passed explicitly (which also skips detection); occasional re-checks
on long finals let a real language change unlock it.

## Table of contents

### Constructors

- [constructor](realtime_transcription.LanguageLock.md#constructor)

### Accessors

- [locked](realtime_transcription.LanguageLock.md#locked)

### Methods

- [languageFor](realtime_transcription.LanguageLock.md#languagefor)
- [observe](realtime_transcription.LanguageLock.md#observe)
- [reset](realtime_transcription.LanguageLock.md#reset)

## Constructors

### constructor

• **new LanguageLock**(`options?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | [`LanguageLockOptions`](../modules/realtime_transcription.md#languagelockoptions) |

#### Defined in

[realtime-transcription/LanguageLock.ts:34](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/LanguageLock.ts#L34)

## Accessors

### locked

• `get` **locked**(): `undefined` \| `string`

The locked language, if any

#### Returns

`undefined` \| `string`

#### Defined in

[realtime-transcription/LanguageLock.ts:46](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/LanguageLock.ts#L46)

## Methods

### languageFor

▸ **languageFor**(`isFinal`, `durationMs`): `undefined` \| `string`

Language to request for a transcription: the locked one, or undefined to
auto-detect (no lock yet, or a final due for a re-check).

#### Parameters

| Name | Type |
| :------ | :------ |
| `isFinal` | `boolean` |
| `durationMs` | `number` |

#### Returns

`undefined` \| `string`

#### Defined in

[realtime-transcription/LanguageLock.ts:54](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/LanguageLock.ts#L54)

___

### observe

▸ **observe**(`language`, `prob`): `void`

A final's auto-detected language (only call it when the language was detected)

#### Parameters

| Name | Type |
| :------ | :------ |
| `language` | `string` |
| `prob` | `undefined` \| `number` |

#### Returns

`void`

#### Defined in

[realtime-transcription/LanguageLock.ts:69](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/LanguageLock.ts#L69)

___

### reset

▸ **reset**(): `void`

#### Returns

`void`

#### Defined in

[realtime-transcription/LanguageLock.ts:95](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/LanguageLock.ts#L95)
