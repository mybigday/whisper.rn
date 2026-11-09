[whisper.rn](../README.md) / [index](../modules/index.md) / ParakeetContext

# Class: ParakeetContext

[index](../modules/index.md).ParakeetContext

## Table of contents

### Constructors

- [constructor](index.ParakeetContext.md#constructor)

### Properties

- [gpu](index.ParakeetContext.md#gpu)
- [id](index.ParakeetContext.md#id)
- [reasonNoGPU](index.ParakeetContext.md#reasonnogpu)

### Methods

- [release](index.ParakeetContext.md#release)
- [transcribe](index.ParakeetContext.md#transcribe)
- [transcribeData](index.ParakeetContext.md#transcribedata)

## Constructors

### constructor

• **new ParakeetContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeParakeetContext` |

#### Defined in

[index.ts:489](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L489)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:485](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L485)

___

### id

• **id**: `number`

#### Defined in

[index.ts:483](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L483)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:487](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L487)

## Methods

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:551](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L551)

___

### transcribe

▸ **transcribe**(`filePathOrBase64`, `options?`): `Object`

Transcribe an audio file path, bundled asset, or base64-encoded WAV.

#### Parameters

| Name | Type |
| :------ | :------ |
| `filePathOrBase64` | `string` \| `number` |
| `options` | [`ParakeetTranscribeOptions`](../modules/index.md#parakeettranscribeoptions) |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> |
| `stop` | () => `Promise`<`void`\> |

#### Defined in

[index.ts:508](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L508)

___

### transcribeData

▸ **transcribeData**(`data`, `options?`): `Object`

Transcribe base64-encoded signed 16-bit PCM data or an ArrayBuffer.

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `string` \| `ArrayBuffer` |
| `options` | [`ParakeetTranscribeOptions`](../modules/index.md#parakeettranscribeoptions) |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> |
| `stop` | () => `Promise`<`void`\> |

#### Defined in

[index.ts:535](https://github.com/mybigday/whisper.rn/blob/2e31d0d/src/index.ts#L535)
