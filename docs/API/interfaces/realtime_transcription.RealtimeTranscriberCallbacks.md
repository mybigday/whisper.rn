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

[realtime-transcription/types.ts:248](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L248)

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

[realtime-transcription/types.ts:250](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L250)

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

[realtime-transcription/types.ts:249](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L249)

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

[realtime-transcription/types.ts:246](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L246)

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

[realtime-transcription/types.ts:247](https://github.com/mybigday/whisper.rn/blob/e931dfc/src/realtime-transcription/types.ts#L247)
