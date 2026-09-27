import { RingBufferVad } from '../RingBufferVad'

describe('RingBufferVad pre-recording window', () => {
  it('keeps only the last preRecordingBufferMs of audio', async () => {
    const sampleRate = 16000
    const preRecordingBufferMs = 1000
    const inferenceIntervalMs = 100
    const bytesPerSecond = sampleRate * 2
    const chunkBytes = Math.floor((inferenceIntervalMs / 1000) * bytesPerSecond)
    const chunks = (2 * bytesPerSecond) / chunkBytes

    let calls = 0
    const vadContext = {
      detectSpeechData: jest.fn(async () => {
        calls += 1
        if (calls < chunks) return []
        // One second of speech, in the 10ms units whisper VAD uses.
        return [{ t0: 0, t1: 100 }]
      }),
    }

    const vad = new RingBufferVad(vadContext as any, {
      sampleRate,
      preRecordingBufferMs,
      inferenceIntervalMs,
      speechRateThreshold: 0.3,
    })

    let started: Uint8Array | null = null
    vad.onSpeechStart((_confidence, data) => {
      started = data
    })

    for (let i = 0; i < chunks; i++) {
      const chunk = new Uint8Array(chunkBytes)
      chunk.fill(i + 1)
      vad.processAudio(chunk)
    }
    await vad.flush()

    expect(started).not.toBeNull()
    // 1000ms at 16kHz, 16-bit mono is 32000 bytes, the second half of the 2s input.
    expect(started!.length).toBe(bytesPerSecond)
    expect(started![0]).toBe(chunks / 2 + 1)
    expect(started![started!.length - 1]).toBe(chunks)
  })
})
