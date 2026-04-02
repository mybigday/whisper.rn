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

[index.ts:235](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L235)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:231](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L231)

___

### id

• **id**: `number`

#### Defined in

[index.ts:229](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L229)

___

### ptr

• **ptr**: `number`

#### Defined in

[index.ts:227](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L227)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:233](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L233)

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

[index.ts:357](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L357)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:363](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L363)

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

[index.ts:265](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L265)

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

[index.ts:318](https://github.com/mybigday/whisper.rn/blob/2d06b36/src/index.ts#L318)
