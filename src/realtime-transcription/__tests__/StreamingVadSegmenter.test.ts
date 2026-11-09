import { StreamingVadSegmenter } from '../StreamingVadSegmenter'
import type { VadOptions } from '../../index'

// Offline references from vad.cpp's vad_segments_from_probs (silero-vad's
// get_speech_timestamps_from_probs): the first two cases are Silero v6.2 on
// jfk.wav, the rest random probability sequences with varied parameters.
const cases: Array<{
  params: Required<Omit<VadOptions, 'maxSpeechDurationS' | 'samplesOverlap'>>
  probs: number[]
  segments: Array<[number, number]>
}> = require('./__fixtures__/vad-segments.json')

const HOP = 512

function runStreaming(
  probs: number[],
  options: VadOptions,
  chunkSizes: () => number,
): Array<[number, number]> {
  const segmenter = new StreamingVadSegmenter(options, HOP)
  const segments: Array<[number, number]> = []
  let open: number | null = null
  const collect = (events: ReturnType<StreamingVadSegmenter['push']>) => {
    events.forEach((event) => {
      if (event.type === 'start') {
        expect(open).toBeNull()
        open = event.sample
      } else {
        expect(open).not.toBeNull()
        segments.push([open!, event.sample])
        open = null
      }
    })
  }
  for (let i = 0; i < probs.length; ) {
    const n = Math.min(chunkSizes(), probs.length - i)
    collect(segmenter.push(probs.slice(i, i + n)))
    i += n
  }
  collect(segmenter.flush(probs.length * HOP))
  expect(open).toBeNull()
  return segments
}

function seededRandom(seed: number) {
  let state = seed
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648
    return state / 2147483648
  }
}

describe('StreamingVadSegmenter', () => {
  const chunkings: Array<[string, () => () => number]> = [
    ['all at once', () => () => Number.MAX_SAFE_INTEGER],
    ['one probability per push', () => () => 1],
    [
      'random pushes',
      () => {
        const random = seededRandom(7)
        return () => 1 + Math.floor(random() * 7)
      },
    ],
  ]

  describe.each(chunkings)('%s', (_name, makeChunker) => {
    it.each(cases.map((c, index) => [index, c] as const))(
      'matches the offline segmentation (case %i)',
      (_index, { params, probs, segments: expected }) => {
        expect(runStreaming(probs, params, makeChunker())).toEqual(expected)
      },
    )
  })

  it('reports a segment only once it is long enough', () => {
    // 250 ms min speech = 7.8 hops of 32 ms
    const segmenter = new StreamingVadSegmenter({ minSpeechDurationMs: 250 }, HOP)
    const starts: number[] = []
    for (let i = 0; i < 20; i += 1) {
      const events = segmenter.push([0.9])
      if (events.some((e) => e.type === 'start')) starts.push(i)
    }
    // Confirmed when the speech known so far exceeds 250 ms: after 8 hops
    expect(starts).toEqual([7])
  })

  it('splits speech longer than maxSpeechDurationS', () => {
    const segmenter = new StreamingVadSegmenter(
      { maxSpeechDurationS: 1, speechPadMs: 30, minSpeechDurationMs: 0 },
      HOP,
    )
    const events = segmenter.push(new Array(100).fill(0.9))
    const ends = events.filter((e) => e.type === 'end').map((e) => e.sample)
    const starts = events.filter((e) => e.type === 'start').map((e) => e.sample)
    expect(ends.length).toBeGreaterThan(0)
    // Consecutive pieces stay contiguous and within the limit
    ends.forEach((end, i) => {
      expect(starts[i + 1]).toBe(end)
      expect(end - starts[i]!).toBeLessThanOrEqual(16000)
    })
  })

  it('splits a short pause between segments instead of reporting an end ahead of the audio', () => {
    // 300 ms pad, 160 ms pause (5 hops): the paddings meet
    const options = { speechPadMs: 300, minSilenceDurationMs: 100, minSpeechDurationMs: 250 }
    const probs = [
      ...new Array(20).fill(0.9),
      ...new Array(5).fill(0.05),
      ...new Array(20).fill(0.9),
      ...new Array(40).fill(0.05),
    ]
    const segmenter = new StreamingVadSegmenter(options, HOP)
    const events: Array<{ type: string; sample: number; seen: number }> = []
    probs.forEach((p) => {
      segmenter.push([p]).forEach((e) =>
        events.push({ type: e.type, sample: e.sample, seen: segmenter.processedSamples }))
    })
    segmenter.flush(probs.length * HOP).forEach((e) =>
      events.push({ type: e.type, sample: e.sample, seen: probs.length * HOP }))

    // Never ahead of the probabilities consumed so far
    events.forEach((e) => expect(e.sample).toBeLessThanOrEqual(e.seen))
    const half = (5 * HOP) / 2
    expect(events.map(({ type, sample }) => [type, sample])).toEqual([
      ['start', 0],
      ['end', 20 * HOP + half],
      ['start', 20 * HOP + half],
      ['end', 45 * HOP + 4800],
    ])
  })

  it('bounds the final partial hop by the real audio length', () => {
    // 7 hops + 16 ms = 240 ms of speech, below the 250 ms minimum
    const segmenter = new StreamingVadSegmenter({ minSpeechDurationMs: 250 }, HOP)
    expect(segmenter.push(new Array(7).fill(0.9))).toEqual([])
    expect(segmenter.flush(7 * HOP + 256, [0.9])).toEqual([])
  })

  it('starts over after reset', () => {
    const segmenter = new StreamingVadSegmenter({ minSpeechDurationMs: 0 }, HOP)
    segmenter.push([0.9, 0.9, 0.9])
    expect(segmenter.inSpeech).toBe(true)
    segmenter.reset()
    expect(segmenter.inSpeech).toBe(false)
    expect(segmenter.processedSamples).toBe(0)
    const events = segmenter.push([0.9])
    expect(events).toEqual([{ type: 'start', sample: 0, prob: 0.9 }])
  })
})
