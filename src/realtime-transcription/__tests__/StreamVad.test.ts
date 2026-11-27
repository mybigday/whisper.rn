import { StreamVad } from '../StreamVad'
import type { StreamVadContextLike, VadStreamLike } from '../StreamVad'
import { StreamingVadSegmenter } from '../StreamingVadSegmenter'

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
  const events: Array<{
    type: string
    samples: Int16Array
    /** start: first sample of `samples`; end: end of the segment */
    sample: number
    reason?: string
  }> = []
  vad.onSpeechStart((_c, data, info) =>
    events.push({ type: 'start', samples: new Int16Array(data.slice().buffer), sample: info.startSample }))
  vad.onSpeechContinue((_c, data, info) =>
    events.push({ type: 'continue', samples: new Int16Array(data.slice().buffer), sample: info.startSample }))
  vad.onSpeechEnd((_c, info) =>
    events.push({ type: 'end', samples: new Int16Array(0), sample: info.endSample, reason: info.reason }))
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

/** Each segment's audio as a consumer keeping positions sees it: chunks in order, cut at the end */
function collectSegments(events: ReturnType<typeof record>): Int16Array[] {
  const segments: Int16Array[] = []
  let current: Int16Array[] = []
  let first = -1
  let next = -1
  events.forEach((e) => {
    if (e.type === 'end') {
      segments.push(concat(current).slice(0, Math.max(0, e.sample - first)))
      current = []
    } else {
      if (e.type === 'start') first = e.sample
      else expect(e.sample).toBe(next) // contiguous
      next = e.sample + e.samples.length
      current.push(e.samples)
    }
  })
  return segments
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

  it('keeps the audio of speech resuming within the padding', async () => {
    const { context } = createFakeContext()
    const vad = new StreamVad(context, {
      vadOptions: { ...options.vadOptions, speechPadMs: 300 },
    })
    const events = record(vad)
    // 160 ms pause: the two segments share it
    const audio = makeAudio([[512, false], [1024, true], [160, false], [1024, true], [1000, false]])
    feed(vad, audio, 100)
    await vad.flush()
    expect(events.filter((e) => e.type === 'start')).toHaveLength(2)
    // Back to back, nothing dropped between them: speech in hops 16-47 and
    // 53-84, padded by 4800 samples at the outer edges
    const emitted = concat(events.map((e) => e.samples))
    expect(emitted).toEqual(audio.slice(16 * HOP - 4800, 85 * HOP + 4800))
  })

  // Each segment's audio must be exactly [start, end) as the segmenter reports
  // it on the same probabilities, so no audio crosses into a neighbour.
  it.each([
    ['pause shorter than 2 * pad', { speechPadMs: 100, minSilenceDurationMs: 50 }, [[1000, false], [1000, true], [150, false], [1000, true], [1000, false]]],
    ['pause between minSilence and 2 * pad', { speechPadMs: 300 }, [[512, false], [1024, true], [160, false], [1024, true], [1000, false]]],
    ['max-duration split', { maxSpeechDurationS: 1 }, [[500, false], [3000, true], [500, false]]],
    ['max-duration split at an earlier pause', { maxSpeechDurationS: 1, minSilenceDurationMs: 300 }, [[300, false], [600, true], [200, false], [600, true], [300, false]]],
  ] as Array<[string, Partial<typeof options.vadOptions>, Array<[number, boolean]>]>)(
    'hands each segment exactly its own audio (%s)',
    async (_name, vadOptions, regions) => {
      const merged = { ...options.vadOptions, ...vadOptions }
      const audio = makeAudio(regions)

      // Same dip choice as StreamVad: mean square of the audio
      const segmenter = new StreamingVadSegmenter(merged, HOP, SR, {
        frameEnergy: (from, to) =>
          audio.slice(from, to).reduce((sum, v) => sum + (v / 32768) ** 2, 0) / (to - from),
      })
      const probs: number[] = []
      for (let i = 0; i < audio.length; i += HOP) {
        const hop = audio.subarray(i, i + HOP)
        probs.push(hop.reduce((sum, v) => sum + Math.abs(v), 0) / HOP > 1000 ? 0.9 : 0.05)
      }
      const lastFull = Math.floor(audio.length / HOP)
      const bounds = [
        ...segmenter.push(probs.slice(0, lastFull)),
        ...segmenter.flush(audio.length, probs.slice(lastFull)),
      ]
      const expected: Int16Array[] = []
      for (let i = 0; i < bounds.length; i += 2) {
        expect(bounds[i]!.type).toBe('start')
        expect(bounds[i + 1]!.type).toBe('end')
        expected.push(audio.slice(bounds[i]!.sample, bounds[i + 1]!.sample))
      }
      expect(expected.length).toBeGreaterThan(1)

      const { context } = createFakeContext()
      const vad = new StreamVad(context, { vadOptions: merged })
      const events = record(vad)
      feed(vad, audio, 10)
      await vad.flush()

      expect(collectSegments(events)).toEqual(expected)
    },
  )

  describe('on the max-split cases (dip splits on)', () => {
    const cases: Array<{ params: any; probs: number[] }> = require('./__fixtures__/vad-segments-max-split.json')

    it.each(cases.map((c, index) => [index, c] as const))(
      'hands each segment exactly its own audio at any chunk size (case %i)',
      async (_index, { params, probs: fixtureProbs }) => {
        // Audio whose fake-context probabilities are 0.9 / 0.05 by the fixture's threshold
        const audio = makeAudio(fixtureProbs.map((p) => [32, p >= params.threshold] as [number, boolean]))
        const vadOptions = { ...params, negThreshold: -1 }
        const segmenter = new StreamingVadSegmenter(vadOptions, HOP, SR, {
          frameEnergy: (from, to) =>
            audio.slice(from, to).reduce((sum, v) => sum + (v / 32768) ** 2, 0) / (to - from),
        })
        const probs = fixtureProbs.map((p) => (p >= params.threshold ? 0.9 : 0.05))
        const bounds = [...segmenter.push(probs), ...segmenter.flush(audio.length)]
        const expected: Int16Array[] = []
        for (let i = 0; i < bounds.length; i += 2) {
          expected.push(audio.slice(bounds[i]!.sample, bounds[i + 1]!.sample))
        }

        // eslint-disable-next-line no-restricted-syntax
        for (const chunkMs of [10, 37]) {
          const { context } = createFakeContext()
          const vad = new StreamVad(context, { vadOptions })
          const events = record(vad)
          feed(vad, audio, chunkMs)
          // eslint-disable-next-line no-await-in-loop
          await vad.flush()
          // Compared as bytes: deep equality over every sample is slow
          const hex = (segments: Int16Array[]) =>
            segments.map((x) => Buffer.from(x.buffer, x.byteOffset, x.byteLength).toString('hex'))
          expect(hex(collectSegments(events))).toEqual(hex(expected))
        }
      },
    )
  })

  it('goes back to an earlier pause at a max-duration split and sends that audio again', async () => {
    const { context } = createFakeContext()
    const vad = new StreamVad(context, {
      vadOptions: { ...options.vadOptions, maxSpeechDurationS: 1, minSilenceDurationMs: 300 },
    })
    const events = record(vad)
    // 600 ms speech, 200 ms pause (shorter than minSilence), 600 ms speech: split at 1 s
    feed(vad, makeAudio([[300, false], [600, true], [200, false], [600, true], [300, false]]), 10)
    await vad.flush()

    const firstEnd = events.findIndex((e) => e.type === 'end')
    const sentBefore = events
      .slice(0, firstEnd)
      .reduce((end, e) => Math.max(end, e.sample + e.samples.length), 0)
    // The end lies within the pause, before audio already handed out
    expect(events[firstEnd]).toMatchObject({ reason: 'max_duration' })
    expect(events[firstEnd]!.sample).toBeGreaterThanOrEqual((900 * SR) / 1000)
    expect(events[firstEnd]!.sample).toBeLessThan((1100 * SR) / 1000)
    expect(events[firstEnd]!.sample).toBeLessThan(sentBefore)
    // The next segment restarts at its padded start, inside that audio
    const next = events[firstEnd + 1]!
    expect(next.type).toBe('start')
    expect(next.sample).toBeLessThan(sentBefore)
    expect(next.sample).toBeGreaterThanOrEqual(events[firstEnd]!.sample)
  })

  it('reports no speech for a final partial hop below the minimum', async () => {
    const { context } = createFakeContext()
    const vad = new StreamVad(context, options)
    const events = record(vad)
    // 7 hops + 16 ms of speech = 240 ms < 250 ms
    feed(vad, makeAudio([[224 + 16, true]]), 100)
    await vad.flush()
    expect(events).toHaveLength(0)
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
