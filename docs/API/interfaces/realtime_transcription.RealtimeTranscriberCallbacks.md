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

[realtime-transcription/types.ts:224](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L224)

___

### onStatsUpdate

• `Optional` **onStatsUpdate**: (`event`: [`RealtimeStatsEvent`](realtime_transcription.RealtimeStatsEvent.md)) => `void`

#### Type declaration

▸ (`event`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `event` | [`RealtimeStatsEvent`](realtime_transcription.RealtimeStatsEvent.md) |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:226](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L226)

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

[realtime-transcription/types.ts:225](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L225)

___

### onTranscribe

• `Optional` **onTranscribe**: (`event`: [`RealtimeTranscribeEvent`](realtime_transcription.RealtimeTranscribeEvent.md)) => `void`

#### Type declaration

▸ (`event`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `event` | [`RealtimeTranscribeEvent`](realtime_transcription.RealtimeTranscribeEvent.md) |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:222](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L222)

___

### onVad

• `Optional` **onVad**: (`event`: [`RealtimeVadEvent`](realtime_transcription.RealtimeVadEvent.md)) => `void`

#### Type declaration

▸ (`event`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `event` | [`RealtimeVadEvent`](realtime_transcription.RealtimeVadEvent.md) |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:223](https://github.com/mybigday/whisper.rn/blob/95a39c1/src/realtime-transcription/types.ts#L223)
