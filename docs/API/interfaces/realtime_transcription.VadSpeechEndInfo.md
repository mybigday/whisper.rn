[whisper.rn](../README.md) / [realtime-transcription](../modules/realtime_transcription.md) / VadSpeechEndInfo

# Interface: VadSpeechEndInfo

[realtime-transcription](../modules/realtime_transcription.md).VadSpeechEndInfo

End of a speech segment

## Table of contents

### Properties

- [endSample](realtime_transcription.VadSpeechEndInfo.md#endsample)
- [reason](realtime_transcription.VadSpeechEndInfo.md#reason)

## Properties

### endSample

• **endSample**: `number`

Stream sample where the segment ends (exclusive). It can lie before audio
already passed for the segment (a max-duration split going back to an
earlier pause); that audio is not part of this segment.

#### Defined in

[realtime-transcription/types.ts:175](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L175)

___

### reason

• **reason**: [`VadSegmentEndReason`](../modules/realtime_transcription.md#vadsegmentendreason)

#### Defined in

[realtime-transcription/types.ts:176](https://github.com/mybigday/whisper.rn/blob/917444a/src/realtime-transcription/types.ts#L176)
