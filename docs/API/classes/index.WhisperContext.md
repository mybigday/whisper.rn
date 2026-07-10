[whisper.rn](../README.md) / [index](../modules/index.md) / WhisperContext

# Class: WhisperContext

[index](../modules/index.md).WhisperContext

## Table of contents

### Constructors

- [constructor](index.WhisperContext.md#constructor)

### Properties

- [gpu](index.WhisperContext.md#gpu)
- [id](index.WhisperContext.md#id)
- [ptr](index.WhisperContext.md#ptr)
- [reasonNoGPU](index.WhisperContext.md#reasonnogpu)

### Methods

- [bench](index.WhisperContext.md#bench)
- [release](index.WhisperContext.md#release)
- [transcribe](index.WhisperContext.md#transcribe)
- [transcribeData](index.WhisperContext.md#transcribedata)

## Constructors

### constructor

• **new WhisperContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeWhisperContext` |

#### Defined in

[index.ts:248](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L248)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:244](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L244)

___

### id

• **id**: `number`

#### Defined in

[index.ts:242](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L242)

___

### ptr

• **ptr**: `number`

#### Defined in

[index.ts:240](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L240)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:246](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L246)

## Methods

### bench

▸ **bench**(`maxThreads`): `Promise`<[`BenchResult`](../modules/index.md#benchresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `maxThreads` | `number` |

#### Returns

`Promise`<[`BenchResult`](../modules/index.md#benchresult)\>

#### Defined in

[index.ts:370](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L370)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:376](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L376)

___

### transcribe

▸ **transcribe**(`filePathOrBase64`, `options?`): `Object`

Transcribe audio file (path or base64 encoded wav file)
base64: need add `data:audio/wav;base64,` prefix

#### Parameters

| Name | Type |
| :------ | :------ |
| `filePathOrBase64` | `string` \| `number` |
| `options` | [`TranscribeFileOptions`](../interfaces/index.TranscribeFileOptions.md) |

#### Returns

`Object`

| Name | Type | Description |
| :------ | :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> | Transcribe result promise |
| `stop` | () => `Promise`<`void`\> | Stop the transcribe |

#### Defined in

[index.ts:278](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L278)

___

### transcribeData

▸ **transcribeData**(`data`, `options?`): `Object`

Transcribe audio data (base64 encoded float32 PCM data or ArrayBuffer)

#### Parameters

| Name | Type |
| :------ | :------ |
| `data` | `string` \| `ArrayBuffer` |
| `options` | [`TranscribeFileOptions`](../interfaces/index.TranscribeFileOptions.md) |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `promise` | `Promise`<[`TranscribeResult`](../modules/index.md#transcriberesult)\> |
| `stop` | () => `Promise`<`void`\> |

#### Defined in

[index.ts:331](https://github.com/mybigday/whisper.rn/blob/9f7d692/src/index.ts#L331)
