import { RealtimeTranscriber } from '../RealtimeTranscriber'
import { SliceManager } from '../SliceManager'
import { JestAudioStreamAdapter } from '../adapters/JestAudioStreamAdapter'

// Regression tests for unbounded memory growth in realtime transcription on slow devices.
// The pre-recording buffer size (RingBufferVad) is fixed separately in #338.

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

describe('SliceManager', () => {
  it('stores data larger than a whole slice without recursing', () => {
    const manager = new SliceManager(1, 10, 16000) // 1 s slices = 32000 bytes
    const size = 32000 * 3 + 100
    expect(() => manager.addAudioData(new Uint8Array(size))).not.toThrow()
    const stored = (manager as any).slices.reduce(
      (sum: number, slice: { sampleCount: number }) => sum + slice.sampleCount,
      0,
    )
    expect(stored).toBe(size)
  })

  it('appends without copying the slice for every chunk', () => {
    const manager = new SliceManager(1, 3, 16000)
    manager.addAudioData(new Uint8Array(3200).fill(1))
    const first = manager.getAudioDataForTranscription(0)!
    manager.addAudioData(new Uint8Array(3200).fill(2))
    const second = manager.getAudioDataForTranscription(0)!
    // Same backing buffer; the earlier view still holds exactly what it held
    expect(second.buffer).toBe(first.buffer)
    expect(first.length).toBe(3200)
    expect(Array.from(first.subarray(0, 4))).toEqual([1, 1, 1, 1])
    expect(second.length).toBe(6400)
  })
})

describe('RealtimeTranscriber transcription queue', () => {
  it('keeps at most one waiting draft per slice while transcription is slow', async () => {
    let release: () => void = () => {}
    const blocked = new Promise<void>((resolve) => {
      release = resolve
    })
    const sentSizes: number[] = []
    const slowWhisper = {
      transcribeData: jest.fn((data: ArrayBuffer) => {
        sentSizes.push(data.byteLength)
        return {
          stop: jest.fn(),
          promise: blocked.then(() => ({ isAborted: false, result: 'text', segments: [] })),
        }
      }),
    }
    const audioStream = new JestAudioStreamAdapter({ chunkSize: 3200, chunkInterval: 100 })
    ;(audioStream as any).startStreaming = jest.fn()
    const transcriber = new RealtimeTranscriber(
      { whisperContext: slowWhisper as any, audioStream },
      { audioSliceSec: 30, realtimeProcessingPauseMs: 1, initRealtimeAfterMs: 1 },
      {},
    )
    await transcriber.start()

    // 3 s of audio, each chunk a trigger for a draft transcription of slice 0
    for (let i = 0; i < 30; i += 1) {
      ;(transcriber as any).handleAudioData({
        data: new Uint8Array(3200),
        sampleRate: 16000,
        channels: 1,
        timestamp: Date.now(),
      })
      // eslint-disable-next-line no-await-in-loop
      await sleep(3)
    }

    // One draft is in flight (blocked); at most one newer draft may wait behind it
    expect((transcriber as any).transcriptionQueue.length).toBeLessThanOrEqual(1)

    // Whisper receives exactly the audio of the slice so far, not the whole preallocated buffer
    expect(sentSizes[0]).toBeLessThan(30 * 16000 * 2)

    release()
    await transcriber.stop()
    await transcriber.release()
  })
})
