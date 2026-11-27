import { StreamingVadSegmenter } from '../StreamingVadSegmenter'
import type { StreamingVadSegmenterOptions, VadSegmenterEvent } from '../StreamingVadSegmenter'
import type { VadOptions } from '../../index'

// Offline references from vad.cpp's vad_segments_from_probs (silero-vad's
// get_speech_timestamps_from_probs): the first two cases are Silero v6.2 on
// jfk.wav, the rest random probability sequences with varied parameters.
const cases: Array<{
  params: Required<Omit<VadOptions, 'maxSpeechDurationS' | 'samplesOverlap'>>
  probs: number[]
  segments: Array<[number, number]>
}> = require('./__fixtures__/vad-segments.json')

// The same, with maxSpeechDurationS: speech-like sequences (long runs, mostly
// short pauses) so that segments get split, at an earlier pause or not.
const maxSplitCases: Array<{
  params: VadOptions
  probs: number[]
  segments: Array<[number, number]>
}> = require('./__fixtures__/vad-segments-max-split.json')

const HOP = 512

function runStreaming(
  probs: number[],
  options: VadOptions,
  chunkSizes: () => number,
  segmenterOptions: StreamingVadSegmenterOptions = {},
): Array<[number, number]> {
  const segmenter = new StreamingVadSegmenter(options, HOP, 16000, segmenterOptions)
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

    it.each(maxSplitCases.map((c, index) => [index, c] as const))(
      'matches the offline max-duration split without dip splits (case %i)',
      (_index, { params, probs, segments: expected }) => {
        expect(runStreaming(probs, params, makeChunker(), { splitAtDip: false })).toEqual(expected)
      },
    )
  })

  describe('default (dip) max-duration split on the max-split cases', () => {
    // Deterministic stand-in for audio energy: louder where the probability is higher
    const energyOf = (probs: number[]) => (start: number) => {
      const p = probs[start / HOP] ?? 0
      return p * p + ((start / HOP) % 7) * 1e-4
    }

    it.each(maxSplitCases.map((c, index) => [index, c] as const))(
      'gives well-formed segments independent of chunking (case %i)',
      (_index, { params, probs }) => {
        const runs = chunkings.map(([, makeChunker]) =>
          runStreaming(probs, params, makeChunker(), { frameEnergy: energyOf(probs) }))
        runs.forEach((segments) => expect(segments).toEqual(runs[0]))

        const limit = (params.maxSpeechDurationS ?? 0) * 16000
        let prevEnd = 0
        runs[0]!.forEach(([start, end]) => {
          expect(start).toBeGreaterThanOrEqual(prevEnd)
          expect(end).toBeGreaterThan(start)
          expect(end - start).toBeLessThanOrEqual(limit)
          prevEnd = end
        })
        expect(prevEnd).toBeLessThanOrEqual(probs.length * HOP)
      },
    )
  })

  it('max-split cases exercise both offline split rules', () => {
    // Guard against a fixture that never splits, or only one way
    let atPause = 0
    let atLimit = 0
    maxSplitCases.forEach(({ params, segments }) => {
      const pad = ((params.speechPadMs ?? 30) * 16000) / 1000
      for (let i = 1; i < segments.length; i += 1) {
        const gap = segments[i]![0] - segments[i - 1]![1]
        const length = segments[i - 1]![1] - segments[i - 1]![0]
        const limit = (params.maxSpeechDurationS ?? 0) * 16000
        if (length > limit - 2 * pad - 2 * HOP && length <= limit) {
          if (gap === 0) atLimit += 1
          else atPause += 1
        }
      }
    })
    expect(atPause).toBeGreaterThan(5)
    expect(atLimit).toBeGreaterThan(0)
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

  describe('max-duration split', () => {
    const run = (
      probs: number[],
      options: VadOptions,
      segmenterOptions: StreamingVadSegmenterOptions = {},
    ) => {
      const segmenter = new StreamingVadSegmenter(options, HOP, 16000, segmenterOptions)
      const events: VadSegmenterEvent[] = []
      probs.forEach((p) => events.push(...segmenter.push([p])))
      events.push(...segmenter.flush(probs.length * HOP))
      return events
    }
    const speech = (n: number) => new Array(n).fill(0.9)
    const quiet = (n: number) => new Array(n).fill(0.05)
    // 1 s limit = 31.25 hops; 0 pad and min speech keep the numbers plain
    const options = {
      maxSpeechDurationS: 1,
      speechPadMs: 0,
      minSpeechDurationMs: 0,
      minSilenceDurationMs: 300,
    }

    it('goes back to the longest earlier pause', () => {
      // pauses of 4 hops (128 ms) at 8 and 7 hops (224 ms) at 18, both shorter than minSilence
      const probs = [...speech(8), ...quiet(4), ...speech(6), ...quiet(7), ...speech(40), ...quiet(20)]
      const events = run(probs, options)
      expect(events.slice(0, 3)).toEqual([
        { type: 'start', sample: 0, prob: 0.9 },
        { type: 'end', sample: 18 * HOP, prob: 0.9, reason: 'max_duration' },
        { type: 'start', sample: 25 * HOP, prob: 0.9 },
      ])
    })

    // A 3 s limit splits at frame 93; the dip window is frames 47-76 (second
    // half of the segment, up to 0.5 s before the split)
    const dipOptions = { ...options, maxSpeechDurationS: 3 }

    it('ignores pauses of 98 ms or less', () => {
      // A 3-hop pause (96 ms): no candidate, so a dip split at its lowest (latest) frame instead
      const probs = [...speech(60), ...quiet(3), ...speech(60), ...quiet(20)]
      const events = run(probs, dipOptions)
      const end = events.find((e) => e.type === 'end')!
      expect(end).toMatchObject({ sample: 62 * HOP + HOP / 2, reason: 'max_duration' })
    })

    it('ends at an ongoing pause longer than every earlier one', () => {
      // Speech up to the limit, then a pause running at the split
      const probs = [...speech(28), ...quiet(5), ...speech(10), ...quiet(20)]
      const events = run(probs, options)
      expect(events.slice(0, 3)).toEqual([
        { type: 'start', sample: 0, prob: 0.9 },
        { type: 'end', sample: 28 * HOP, prob: 0.05, reason: 'max_duration' },
        { type: 'start', sample: 33 * HOP, prob: 0.9 },
      ])
    })

    it('splits continuous speech at the least speech-like frame', () => {
      const probs = speech(150)
      probs[58] = 0.6
      probs[60] = 0.55 // the lowest
      const events = run(probs, dipOptions)
      expect(events.slice(0, 3)).toEqual([
        { type: 'start', sample: 0, prob: 0.9 },
        { type: 'end', sample: 60 * HOP + HOP / 2, prob: 0.9, reason: 'max_duration', inSpeech: true },
        { type: 'start', sample: 60 * HOP + HOP / 2, prob: 0.55 },
      ])
    })

    it('picks the quietest of the near-lowest frames', () => {
      const probs = speech(150)
      probs[58] = 0.6 // near the lowest, but the quietest
      probs[60] = 0.58
      const frameEnergy = (start: number) => (start === 58 * HOP ? 0.001 : 0.1)
      const events = run(probs, dipOptions, { frameEnergy })
      expect(events[1]).toMatchObject({ type: 'end', sample: 58 * HOP + HOP / 2 })
    })

    it('never picks a frame within 0.5 s of the limit or in the first half', () => {
      const probs = speech(150)
      probs[85] = 0.1 // lowest, but in the last 0.5 s before the split
      probs[20] = 0.1 // lowest, but in the first half
      probs[60] = 0.6
      const events = run(probs, dipOptions)
      expect(events[1]).toMatchObject({ type: 'end', sample: 60 * HOP + HOP / 2 })
    })

    it('does not split again right after a split', () => {
      // Energy rising with time makes the earliest window frame the quietest
      const events = run(speech(400), dipOptions, { frameEnergy: (start) => start })
      const ends = events.filter((e) => e.type === 'end').map((e) => e.sample)
      ends.forEach((end, i) => {
        if (i > 0) expect(end - ends[i - 1]!).toBeGreaterThanOrEqual(1.5 * 16000 - HOP)
      })
    })

    it('cuts at the limit like offline without dip splits', () => {
      const probs = speech(60)
      probs[10] = 0.6
      const events = run(probs, options, { splitAtDip: false })
      // Splits at frame 31 (the first one past 31.25 - 1 hops); the next segment starts at frame 32
      expect(events.slice(0, 3)).toEqual([
        { type: 'start', sample: 0, prob: 0.9 },
        { type: 'end', sample: 31 * HOP, prob: 0.9, reason: 'max_duration', inSpeech: true },
        { type: 'start', sample: 32 * HOP, prob: 0.9 },
      ])
    })

    it('keeps the whole open segment while it may be split back', () => {
      const segmenter = new StreamingVadSegmenter(options, HOP)
      segmenter.push(speech(20))
      expect(segmenter.maySplitBack).toBe(true)
      expect(segmenter.retainFrom).toBe(0)
    })
  })

  it('reports why a segment ended', () => {
    const segmenter = new StreamingVadSegmenter({ minSpeechDurationMs: 0, speechPadMs: 0 }, HOP)
    const events = [
      // 6 quiet hops (192 ms) exceed the 100 ms minimum silence
      ...segmenter.push([0.9, 0.9, 0.9, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9, 0.9]),
      ...segmenter.flush(11 * HOP),
    ]
    expect(events.filter((e) => e.type === 'end').map((e) => (e as { reason: string }).reason))
      .toEqual(['pause', 'end_of_stream'])
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
