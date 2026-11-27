[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / VadSpeechEndInfo

# Interface: VadSpeechEndInfo

[realtime-transcription](../modules/realtime_transcription.md).VadSpeechEndInfo

End of a speech segment

## Table of contents

### Properties

- [endSample](realtime_transcription.VadSpeechEndInfo.md#endsample)
- [inSpeech](realtime_transcription.VadSpeechEndInfo.md#inspeech)
- [reason](realtime_transcription.VadSpeechEndInfo.md#reason)

## Properties

### endSample

• **endSample**: `number`

Stream sample where the segment ends (exclusive). It can lie before audio
already passed for the segment (a max-duration split going back to an
earlier pause); that audio is not part of this segment.

#### Defined in

[realtime-transcription/types.ts:177](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L177)

___

### inSpeech

• `Optional` **inSpeech**: `boolean`

A max-duration split inside speech: the next segment starts at `endSample`

#### Defined in

[realtime-transcription/types.ts:180](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L180)

___

### reason

• **reason**: [`VadSegmentEndReason`](../modules/realtime_transcription.md#vadsegmentendreason)

#### Defined in

[realtime-transcription/types.ts:178](https://github.com/mybigday/whisper.rn/blob/90f6961/src/realtime-transcription/types.ts#L178)
