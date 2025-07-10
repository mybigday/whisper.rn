[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / AudioStreamInterface

# Interface: AudioStreamInterface

[realtime-transcription](../modules/realtime_transcription.md).AudioStreamInterface

## Table of contents

### Methods

- [initialize](realtime_transcription.AudioStreamInterface.md#initialize)
- [isRecording](realtime_transcription.AudioStreamInterface.md#isrecording)
- [onData](realtime_transcription.AudioStreamInterface.md#ondata)
- [onError](realtime_transcription.AudioStreamInterface.md#onerror)
- [onStatusChange](realtime_transcription.AudioStreamInterface.md#onstatuschange)
- [release](realtime_transcription.AudioStreamInterface.md#release)
- [start](realtime_transcription.AudioStreamInterface.md#start)
- [stop](realtime_transcription.AudioStreamInterface.md#stop)

## Methods

### initialize

▸ **initialize**(`config`): `Promise`<`void`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `config` | [`AudioStreamConfig`](realtime_transcription.AudioStreamConfig.md) |

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:22](https://github.com/mybigday/whisper.rn/blob/16b3c27/src/realtime-transcription/types.ts#L22)

___

### isRecording

▸ **isRecording**(): `boolean`

#### Returns

`boolean`

#### Defined in

[realtime-transcription/types.ts:25](https://github.com/mybigday/whisper.rn/blob/16b3c27/src/realtime-transcription/types.ts#L25)

___

### onData

▸ **onData**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`data`: [`AudioStreamData`](realtime_transcription.AudioStreamData.md)) => `void` |

#### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:26](https://github.com/mybigday/whisper.rn/blob/16b3c27/src/realtime-transcription/types.ts#L26)

___

### onError

▸ **onError**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`error`: `string`) => `void` |

#### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:27](https://github.com/mybigday/whisper.rn/blob/16b3c27/src/realtime-transcription/types.ts#L27)

___

### onStatusChange

▸ **onStatusChange**(`callback`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `callback` | (`isRecording`: `boolean`) => `void` |

#### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:28](https://github.com/mybigday/whisper.rn/blob/16b3c27/src/realtime-transcription/types.ts#L28)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:29](https://github.com/mybigday/whisper.rn/blob/16b3c27/src/realtime-transcription/types.ts#L29)

___

### start

▸ **start**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:23](https://github.com/mybigday/whisper.rn/blob/16b3c27/src/realtime-transcription/types.ts#L23)

___

### stop

▸ **stop**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[realtime-transcription/types.ts:24](https://github.com/mybigday/whisper.rn/blob/16b3c27/src/realtime-transcription/types.ts#L24)
