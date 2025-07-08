[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeTranscriberCallbacks

# Interface: RealtimeTranscriberCallbacks

[realtime-transcription](../modules/realtime_transcription.md).RealtimeTranscriberCallbacks

## Table of contents

### Properties

- [onError](realtime_transcription.RealtimeTranscriberCallbacks.md#onerror)
- [onStatsUpdate](realtime_transcription.RealtimeTranscriberCallbacks.md#onstatsupdate)
- [onStatusChange](realtime_transcription.RealtimeTranscriberCallbacks.md#onstatuschange)
- [onTranscribe](realtime_transcription.RealtimeTranscriberCallbacks.md#ontranscribe)
- [onVad](realtime_transcription.RealtimeTranscriberCallbacks.md#onvad)

## Properties

### onError

• `Optional` **onError**: (`error`: `string`) => `void`

#### Type declaration

▸ (`error`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `error` | `string` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:225](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L225)

___

### onStatsUpdate

• `Optional` **onStatsUpdate**: (`event`: [`StatsEvent`](realtime_transcription.StatsEvent.md)) => `void`

#### Type declaration

▸ (`event`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `event` | [`StatsEvent`](realtime_transcription.StatsEvent.md) |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:227](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L227)

___

### onStatusChange

• `Optional` **onStatusChange**: (`isActive`: `boolean`) => `void`

#### Type declaration

▸ (`isActive`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `isActive` | `boolean` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:226](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L226)

___

### onTranscribe

• `Optional` **onTranscribe**: (`event`: [`TranscribeEvent`](realtime_transcription.TranscribeEvent.md)) => `void`

#### Type declaration

▸ (`event`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `event` | [`TranscribeEvent`](realtime_transcription.TranscribeEvent.md) |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:223](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L223)

___

### onVad

• `Optional` **onVad**: (`event`: [`VadEvent`](realtime_transcription.VadEvent.md)) => `void`

#### Type declaration

▸ (`event`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `event` | [`VadEvent`](realtime_transcription.VadEvent.md) |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:224](https://github.com/mybigday/whisper.rn/blob/5c1c70c/src/realtime-transcription/types.ts#L224)
