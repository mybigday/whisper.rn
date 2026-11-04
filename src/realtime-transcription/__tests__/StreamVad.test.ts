import { StreamVad } from '../StreamVad'
import type { StreamVadContextLike, VadStreamLike } from '../StreamVad'

const SR = 16000
const HOP = 512

// A stand-in for WhisperVadStream: one probability per hop, 0.9 for loud audio.
function createFakeContext() {
  const pushes: number[] = []
  let pending: number[] = []
  const toProbs = (final: boolean) => {
    const probs: number[] = []
    while (pending.length >= HOP || (final && pending.length > 0)) {
      const hop = pending.splice(0, HOP)
      const mean = hop.reduce((sum, v) => sum + Math.abs(v), 0) / HOP
      probs.push(mean > 1000 ? 0.9 : 0.05)
    }
    return probs
  }
  const stream: VadStreamLike = {
    push: jest.fn(async (audio: Uint8Array) => {
      pushes.push(audio.byteLength / 2)
      const samples = new Int16Array(audio.buffer, audio.byteOffset, audio.byteLength / 2)
      pending.push(...Array.from(samples))
      return toProbs(false)
    }),
    flush: jest.fn(async () => toProbs(true)),
    reset: jest.fn(async () => {
      pending = []
    }),
    release: jest.fn(async () => undefined),
  }
  const context: StreamVadContextLike = {
    hopSamples: HOP,
    sampleRate: SR,
    createStream: jest.fn(async () => stream),
  }
  return { context, stream, pushes }
}

/** PCM16 with loud (speech) and quiet regions, sample value = its index mod 30000 + offset */
function makeAudio(regions: Array<[number, boolean]>): Int16Array {
  const total = regions.reduce((n, [ms]) => n + (ms * SR) / 1000, 0)
  const audio = new Int16Array(total)
  let i = 0
  regions.forEach(([ms, loud]) => {
    const n = (ms * SR) / 1000
    for (let k = 0; k < n; k += 1, i += 1) {
      // Tag every sample with its position so the emitted audio can be checked
      const tag = i % 1000
      audio[i] = loud ? 2000 + tag : tag - 500
    }
  })
  return audio
}

function feed(vad: StreamVad, audio: Int16Array, chunkMs: number) {
  const chunk = (chunkMs * SR) / 1000
  for (let i = 0; i < audio.length; i += chunk) {
    const part = audio.slice(i, Math.min(i + chunk, audio.length))
    vad.processAudio(new Uint8Array(part.buffer))
  }
}

function record(vad: StreamVad) {
  const events: Array<{ type: string; samples: Int16Array }> = []
  vad.onSpeechStart((_c, data) =>
    events.push({ type: 'start', samples: new Int16Array(data.slice().buffer) }))
  vad.onSpeechContinue((_c, data) =>
    events.push({ type: 'continue', samples: new Int16Array(data.slice().buffer) }))
  vad.onSpeechEnd(() => events.push({ type: 'end', samples: new Int16Array(0) }))
  return events
}

function concat(parts: Int16Array[]): Int16Array {
  const out = new Int16Array(parts.reduce((n, p) => n + p.length, 0))
  let offset = 0
  parts.forEach((p) => {
    out.set(p, offset)
    offset += p.length
  })
  return out
}

describe('StreamVad', () => {
  const options = {
    vadOptions: {
      threshold: 0.5,
      minSpeechDurationMs: 250,
      minSilenceDurationMs: 100,
      speechPadMs: 30,
      maxSpeechDurationS: 0,
    },
  }

  it('pushes each chunk once and emits the padded segment audio exactly once', async () => {
    const { context, pushes } = createFakeContext()
    const vad = new StreamVad(context, options)
    const events = record(vad)
    // 1 s silence, 2 s speech, 1 s silence (speech starts/ends on hop boundaries
    // 16000 / 512 is not whole, so the edges fall inside hops)
    const audio = makeAudio([[1000, false], [2000, true], [1000, false]])
    feed(vad, audio, 100)
    await vad.flush()

    expect(pushes.length).toBe(40)
    expect(pushes.every((n) => n === 1600)).toBe(true)

    expect(events[0]!.type).toBe('start')
    expect(events[events.length - 1]!.type).toBe('end')
    expect(events.filter((e) => e.type === 'start')).toHaveLength(1)
    expect(events.filter((e) => e.type === 'end')).toHaveLength(1)

    // First hop whose mean is loud starts the speech; the segment spans
    // [start - pad, pauseStart + pad)
    const pad = 480
    const speechStartHop = Math.ceil((SR * 1 - HOP / 2) / HOP)
    const pauseStartHop = Math.ceil((SR * 3 - HOP / 2) / HOP)
    const from = speechStartHop * HOP - pad
    const to = pauseStartHop * HOP + pad
    const emitted = concat(events.map((e) => e.samples))
    expect(emitted).toEqual(audio.slice(from, to))
  })

  it('does not report speech shorter than minSpeechDurationMs', async () => {
    const { context } = createFakeContext()
    const vad = new StreamVad(context, options)
    const events = record(vad)
    feed(vad, makeAudio([[500, false], [150, true], [1000, false]]), 100)
    await vad.flush()
    expect(events).toHaveLength(0)
  })

  it('reports speech still open at flush up to the end of the audio', async () => {
    const { context } = createFakeContext()
    const vad = new StreamVad(context, options)
    const events = record(vad)
    const audio = makeAudio([[500, false], [1000, true]])
    feed(vad, audio, 100)
    await vad.flush()
    expect(events.map((e) => e.type)[0]).toBe('start')
    expect(events[events.length - 1]!.type).toBe('end')
    const emitted = concat(events.map((e) => e.samples))
    expect(emitted[emitted.length - 1]).toBe(audio[audio.length - 1])
  })

  it('reset clears state; audio after a flush starts a new stream', async () => {
    const { context, stream } = createFakeContext()
    const vad = new StreamVad(context, options)
    const events = record(vad)
    feed(vad, makeAudio([[300, false], [600, true]]), 100)
    await vad.flush()
    feed(vad, makeAudio([[300, false], [600, true]]), 100)
    await vad.flush()
    expect(stream.reset).toHaveBeenCalledTimes(1)
    expect(events.filter((e) => e.type === 'start')).toHaveLength(2)

    await vad.reset()
    expect(stream.reset).toHaveBeenCalledTimes(2)
    await vad.release()
    expect(stream.release).toHaveBeenCalledTimes(1)
  })

  it('reports push failures through onError and keeps going', async () => {
    const { context, stream } = createFakeContext()
    const vad = new StreamVad(context, options)
    const errors: string[] = []
    vad.onError((e) => errors.push(e))
    ;(stream.push as jest.Mock).mockRejectedValueOnce(new Error('boom'))
    feed(vad, makeAudio([[200, false]]), 100)
    await vad.flush()
    expect(errors).toEqual(['VAD processing error: boom'])
    expect(stream.push).toHaveBeenCalledTimes(2)
  })
})
