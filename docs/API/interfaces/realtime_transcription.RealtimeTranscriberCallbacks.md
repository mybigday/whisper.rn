[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / RealtimeTranscriberCallbacks

# Interface: RealtimeTranscriberCallbacks

[realtime-transcription](../modules/realtime_transcription.md).RealtimeTranscriberCallbacks

## Table of contents

### Properties

- [onBeginTranscribe](realtime_transcription.RealtimeTranscriberCallbacks.md#onbegintranscribe)
- [onBeginVad](realtime_transcription.RealtimeTranscriberCallbacks.md#onbeginvad)
- [onError](realtime_transcription.RealtimeTranscriberCallbacks.md#onerror)
- [onSliceTranscriptionStabilized](realtime_transcription.RealtimeTranscriberCallbacks.md#onslicetranscriptionstabilized)
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

[realtime-transcription/types.ts:243](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/types.ts#L243)

___

### onBeginVad

• `Optional` **onBeginVad**: (`sliceInfo`: { `audioData`: `Uint8Array` ; `duration`: `number` ; `sliceIndex`: `number`  }) => `Promise`<`boolean`\>

#### Type declaration

▸ (`sliceInfo`): `Promise`<`boolean`\>

##### Parameters

| Name | Type |
| :------ | :------ |
| `sliceInfo` | `Object` |
| `sliceInfo.audioData` | `Uint8Array` |
| `sliceInfo.duration` | `number` |
| `sliceInfo.sliceIndex` | `number` |

##### Returns

`Promise`<`boolean`\>

#### Defined in

[realtime-transcription/types.ts:250](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/types.ts#L250)

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

[realtime-transcription/types.ts:256](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/types.ts#L256)

___

### onSliceTranscriptionStabilized

• `Optional` **onSliceTranscriptionStabilized**: (`text`: `string`) => `void`

#### Type declaration

▸ (`text`): `void`

##### Parameters

| Name | Type |
| :------ | :------ |
| `text` | `string` |

##### Returns

`void`

#### Defined in

[realtime-transcription/types.ts:259](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/types.ts#L259)

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

[realtime-transcription/types.ts:258](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/types.ts#L258)

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

[realtime-transcription/types.ts:257](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/types.ts#L257)

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

[realtime-transcription/types.ts:249](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/types.ts#L249)

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

[realtime-transcription/types.ts:255](https://github.com/mybigday/whisper.rn/blob/25a2438/src/realtime-transcription/types.ts#L255)
