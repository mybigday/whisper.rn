[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / WavFileWriterFs

# Interface: WavFileWriterFs

[realtime-transcription](../modules/realtime_transcription.md).WavFileWriterFs

## Table of contents

### Properties

- [appendFile](realtime_transcription.WavFileWriterFs.md#appendfile)
- [exists](realtime_transcription.WavFileWriterFs.md#exists)
- [readFile](realtime_transcription.WavFileWriterFs.md#readfile)
- [unlink](realtime_transcription.WavFileWriterFs.md#unlink)
- [writeFile](realtime_transcription.WavFileWriterFs.md#writefile)

## Properties

### appendFile

• **appendFile**: (`filePath`: `string`, `data`: `string`, `encoding`: `string`) => `Promise`<`void`\>

#### Type declaration

▸ (`filePath`, `data`, `encoding`): `Promise`<`void`\>

##### Parameters

| Name | Type |
| :------ | :------ |
| `filePath` | `string` |
| `data` | `string` |
| `encoding` | `string` |

##### Returns

`Promise`<`void`\>

#### Defined in

[utils/WavFileWriter.ts:11](https://github.com/mybigday/whisper.rn/blob/0152db5/src/utils/WavFileWriter.ts#L11)

___

### exists

• **exists**: (`filePath`: `string`) => `Promise`<`boolean`\>

#### Type declaration

▸ (`filePath`): `Promise`<`boolean`\>

##### Parameters

| Name | Type |
| :------ | :------ |
| `filePath` | `string` |

##### Returns

`Promise`<`boolean`\>

#### Defined in

[utils/WavFileWriter.ts:13](https://github.com/mybigday/whisper.rn/blob/0152db5/src/utils/WavFileWriter.ts#L13)

___

### readFile

• **readFile**: (`filePath`: `string`, `encoding`: `string`) => `Promise`<`string`\>

#### Type declaration

▸ (`filePath`, `encoding`): `Promise`<`string`\>

##### Parameters

| Name | Type |
| :------ | :------ |
| `filePath` | `string` |
| `encoding` | `string` |

##### Returns

`Promise`<`string`\>

#### Defined in

[utils/WavFileWriter.ts:12](https://github.com/mybigday/whisper.rn/blob/0152db5/src/utils/WavFileWriter.ts#L12)

___

### unlink

• **unlink**: (`filePath`: `string`) => `Promise`<`void`\>

#### Type declaration

▸ (`filePath`): `Promise`<`void`\>

##### Parameters

| Name | Type |
| :------ | :------ |
| `filePath` | `string` |

##### Returns

`Promise`<`void`\>

#### Defined in

[utils/WavFileWriter.ts:14](https://github.com/mybigday/whisper.rn/blob/0152db5/src/utils/WavFileWriter.ts#L14)

___

### writeFile

• **writeFile**: (`filePath`: `string`, `data`: `string`, `encoding`: `string`) => `Promise`<`void`\>

#### Type declaration

▸ (`filePath`, `data`, `encoding`): `Promise`<`void`\>

##### Parameters

| Name | Type |
| :------ | :------ |
| `filePath` | `string` |
| `data` | `string` |
| `encoding` | `string` |

##### Returns

`Promise`<`void`\>

#### Defined in

[utils/WavFileWriter.ts:10](https://github.com/mybigday/whisper.rn/blob/0152db5/src/utils/WavFileWriter.ts#L10)
