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

[index.ts:481](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/index.ts#L481)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:477](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/index.ts#L477)

___

### id

• **id**: `number`

#### Defined in

[index.ts:475](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/index.ts#L475)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:479](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/index.ts#L479)

## Methods

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:543](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/index.ts#L543)

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

[index.ts:500](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/index.ts#L500)

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

[index.ts:527](https://github.com/mybigday/whisper.rn/blob/db23f7b/src/index.ts#L527)
