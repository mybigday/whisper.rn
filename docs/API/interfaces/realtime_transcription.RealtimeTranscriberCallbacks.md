[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeTranscriberCallbacks

# Interface: RealtimeTranscriberCallbacks

[realtime-transcription](../modules/realtime_transcription.md).RealtimeTranscriberCallbacks

## Table of contents

### Properties

- [onBeginTranscribe](realtime_transcription.RealtimeTranscriberCallbacks.md#onbegintranscribe)
- [onError](realtime_transcription.RealtimeTranscriberCallbacks.md#onerror)
- [onStatsUpdate](realtime_transcription.RealtimeTranscriberCallbacks.md#onstatsupdate)
- [onStatusChange](realtime_transcription.RealtimeTranscriberCallbacks.md#onstatuschange)
- [onTranscribe](realtime_transcription.RealtimeTranscriberCallbacks.md#ontranscribe)
- [onVad](realtime_transcription.RealtimeTranscriberCallbacks.md#onvad)

## Properties

### onBeginTranscribe

• `Optional` **onBeginTranscribe**: (`sliceInfo`: { `audioData`: `Uint8Array` ; `duration`: `number` ; `sliceIndex`: `number` ; `vadEvent?`: [`RealtimeVadEvent`](realtime_transcription.RealtimeVadEvent.md)  }) => `Promise`<`boolean`\>

#### Type declaration

▸ (`sliceInfo`): `Promise`<`boolean`\>

##### Parameters

| Name | Type |
| :------ | :------ |
| `sliceInfo` | `Object` |
| `sliceInfo.audioData` | `Uint8Array` |
| `sliceInfo.duration` | `number` |
| `sliceInfo.sliceIndex` | `number` |
| `sliceInfo.vadEvent?` | [`RealtimeVadEvent`](realtime_transcription.RealtimeVadEvent.md) |

##### Returns

`Promise`<`boolean`\>

#### Defined in

[realtime-transcription/types.ts:246](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L246)

___

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

[realtime-transcription/types.ts:254](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L254)

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

[realtime-transcription/types.ts:256](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L256)

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

[realtime-transcription/types.ts:255](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L255)

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

[realtime-transcription/types.ts:252](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L252)

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

[realtime-transcription/types.ts:253](https://github.com/mybigday/whisper.rn/blob/ee85d12/src/realtime-transcription/types.ts#L253)
