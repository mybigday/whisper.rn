[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RingBufferVad

# Class: RingBufferVad

[realtime-transcription](../modules/realtime_transcription.md).RingBufferVad

## Implements

- `RealtimeVadContextLike`

## Table of contents

### Constructors

- [constructor](realtime_transcription.RingBufferVad.md#constructor)

### Methods

- [flush](realtime_transcription.RingBufferVad.md#flush)
- [onError](realtime_transcription.RingBufferVad.md#onerror)
- [onSpeechContinue](realtime_transcription.RingBufferVad.md#onspeechcontinue)
- [onSpeechEnd](realtime_transcription.RingBufferVad.md#onspeechend)
- [onSpeechStart](realtime_transcription.RingBufferVad.md#onspeechstart)
- [processAudio](realtime_transcription.RingBufferVad.md#processaudio)
- [reset](realtime_transcription.RingBufferVad.md#reset)
- [updateOptions](realtime_transcription.RingBufferVad.md#updateoptions)

## Constructors

### constructor

• **new RingBufferVad**(`vadContext`, `options?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `vadContext` | `WhisperVadContextLike` |
| `options` | `RingBufferVadOptions` |

#### Defined in

[realtime-transcription/RingBufferVad.ts:51](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L51)

## Methods

### flush

▸ **flush**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Implementation of

RealtimeVadContextLike.flush

#### Defined in

[realtime-transcription/RingBufferVad.ts:132](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L132)

___

### onError

▸ **onError**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`error`: `string`) => `void` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.onError

#### Defined in

[realtime-transcription/RingBufferVad.ts:98](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L98)

___

### onSpeechContinue

▸ **onSpeechContinue**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `data`: `Uint8Array`) => `void` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.onSpeechContinue

#### Defined in

[realtime-transcription/RingBufferVad.ts:90](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L90)

___

### onSpeechEnd

▸ **onSpeechEnd**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`) => `void` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.onSpeechEnd

#### Defined in

[realtime-transcription/RingBufferVad.ts:94](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L94)

___

### onSpeechStart

▸ **onSpeechStart**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`confidence`: `number`, `data`: `Uint8Array`) => `void` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.onSpeechStart

#### Defined in

[realtime-transcription/RingBufferVad.ts:86](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L86)

___

### processAudio

▸ **processAudio**(`data`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `Uint8Array` |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.processAudio

#### Defined in

[realtime-transcription/RingBufferVad.ts:102](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L102)

___

### reset

▸ **reset**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Implementation of

RealtimeVadContextLike.reset

#### Defined in

[realtime-transcription/RingBufferVad.ts:155](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L155)

___

### updateOptions

▸ **updateOptions**(`options`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Partial`<[`VadOptions`](../modules/index.md#vadoptions)\> |

#### Returns

`void`

#### Implementation of

RealtimeVadContextLike.updateOptions

#### Defined in

[realtime-transcription/RingBufferVad.ts:280](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/realtime-transcription/RingBufferVad.ts#L280)
