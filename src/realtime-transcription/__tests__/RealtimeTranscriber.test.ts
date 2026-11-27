import { RealtimeTranscriber, isRepetitive } from '../RealtimeTranscriber'
import { StreamVad } from '../StreamVad'
import type { StreamVadContextLike, VadStreamLike } from '../StreamVad'
import { StreamingVadSegmenter } from '../StreamingVadSegmenter'
import { JestAudioStreamAdapter } from '../adapters/JestAudioStreamAdapter'
import type { RealtimeTranscribeEvent } from '../types'
import type { ParakeetTranscribeOptions, TranscribeResult } from '../../index'

// Mock WavFileWriter
const mockWavFileWriter = {
  initialize: jest.fn().mockResolvedValue(undefined),
  appendAudioData: jest.fn().mockResolvedValue(undefined),
  finalize: jest.fn().mockResolvedValue(undefined),
  cancel: jest.fn().mockResolvedValue(undefined),
}

jest.mock('../../utils/WavFileWriter', () => ({
  WavFileWriter: jest.fn().mockImplementation(() => mockWavFileWriter),
}))

const SR = 16000
const HOP = 512

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** Let the segmentation and transcription queues run */
const settle = async () => {
  for (let i = 0; i < 10; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    await sleep(1)
  }
}

/** PCM16 with loud (speech) and quiet regions */
function makeAudio(regions: Array<[number, boolean]>): Int16Array {
  const total = regions.reduce((n, [ms]) => n + (ms * SR) / 1000, 0)
  const audio = new Int16Array(total)
  let i = 0
  regions.forEach(([ms, loud]) => {
    const n = (ms * SR) / 1000
    for (let k = 0; k < n; k += 1, i += 1) {
      const tag = i % 1000
      audio[i] = loud ? 2000 + tag : tag - 500
    }
  })
  return audio
}

const bytesOf = (audio: Int16Array) => new Uint8Array(audio.buffer, audio.byteOffset, audio.byteLength)

/** A probability source standing in for WhisperVadContext: 0.9 for loud hops */
function createFakeVad(): StreamVadContextLike {
  return {
    hopSamples: HOP,
    sampleRate: SR,
    createStream: async (): Promise<VadStreamLike> => {
      let pending: number[] = []
      const toProbs = (final: boolean) => {
        const probs: number[] = []
        while (pending.length >= HOP || (final && pending.length > 0)) {
          const hop = pending.splice(0, HOP)
          probs.push(hop.reduce((s, v) => s + Math.abs(v), 0) / hop.length > 1000 ? 0.9 : 0.05)
        }
        return probs
      }
      return {
        push: async (audio: Uint8Array) => {
          const samples = new Int16Array(audio.slice().buffer)
          pending.push(...Array.from(samples))
          return toProbs(false)
        },
        flush: async () => toProbs(true),
        reset: async () => {
          pending = []
        },
        release: async () => undefined,
      }
    },
  }
}

type Call = { samples: Int16Array; options: any }

/** A Whisper stand-in recording each request; `reply` builds the result */
function createWhisper(
  reply: (call: Call, index: number) => Partial<TranscribeResult> = (_c, i) => ({ result: `t${i}` }),
) {
  const calls: Call[] = []
  const context = {
    transcribeData: jest.fn((data: ArrayBuffer, options: any) => {
      const call = { samples: new Int16Array(data.slice(0)), options }
      calls.push(call)
      const result = {
        isAborted: false,
        language: 'en',
        result: '',
        segments: [],
        ...reply(call, calls.length - 1),
      } as TranscribeResult
      if (result.segments.length === 0 && result.result) {
        result.segments = [{ text: result.result, t0: 0, t1: 100 }]
      }
      return { stop: jest.fn().mockResolvedValue(undefined), promise: Promise.resolve(result) }
    }),
  }
  return { context, calls }
}

function createStream() {
  const stream = new JestAudioStreamAdapter({ chunkSize: 3200, chunkInterval: 100, generateSilence: false })
  ;(stream as any).startStreaming = jest.fn()
  return stream
}

/** Feed audio in chunks, letting the pipeline run after each */
async function feed(stream: JestAudioStreamAdapter, audio: Int16Array, chunkMs = 100) {
  const chunk = (chunkMs * SR) / 1000
  for (let i = 0; i < audio.length; i += chunk) {
    stream.simulateDataChunk(bytesOf(audio.slice(i, Math.min(i + chunk, audio.length))))
    // eslint-disable-next-line no-await-in-loop
    await settle()
  }
}

const transcribeEvents = (onTranscribe: jest.Mock): RealtimeTranscribeEvent[] =>
  onTranscribe.mock.calls.map(([e]) => e).filter((e: RealtimeTranscribeEvent) => e.type === 'transcribe')

const finalsOf = (onTranscribe: jest.Mock) => transcribeEvents(onTranscribe).filter((e) => e.isFinal)

describe('RealtimeTranscriber', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('construction', () => {
    it('requires a transcription context', () => {
      expect(() => new RealtimeTranscriber({ audioStream: createStream() } as any)).toThrow(
        /whisperContext or parakeetContext/,
      )
    })

    it('takes a segmenter VAD as it is and recommends the probability source', () => {
      const logger = jest.fn()
      const segmenter = new StreamVad(createFakeVad())
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, vadContext: segmenter, audioStream: createStream() },
        { logger },
      )
      expect(t.getStatistics().vadEnabled).toBe(true)
      expect(logger).toHaveBeenCalledWith(expect.stringContaining('Pass the probability source instead'))
    })

    it('reports whether a VAD is used', () => {
      const { context } = createWhisper()
      const withVad = new RealtimeTranscriber({
        whisperContext: context,
        vadContext: createFakeVad(),
        audioStream: createStream(),
      })
      const without = new RealtimeTranscriber({ whisperContext: context, audioStream: createStream() })
      expect(withVad.getStatistics().vadEnabled).toBe(true)
      expect(without.getStatistics().vadEnabled).toBe(false)
    })
  })

  describe('lifecycle', () => {
    it('starts and stops, initializing and stopping the audio stream', async () => {
      const stream = createStream()
      const initialize = jest.spyOn(stream, 'initialize')
      const stop = jest.spyOn(stream, 'stop')
      const onStatusChange = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, audioStream: stream },
        { audioStreamConfig: { sampleRate: 16000, channels: 1, bitsPerSample: 16 } },
        { onStatusChange },
      )
      await t.start()
      expect(t.getStatistics().isActive).toBe(true)
      expect(initialize).toHaveBeenCalledWith(expect.objectContaining({ sampleRate: 16000, channels: 1 }))
      await expect(t.start()).rejects.toThrow('already active')
      await t.stop()
      expect(t.getStatistics().isActive).toBe(false)
      expect(stop).toHaveBeenCalled()
      expect(onStatusChange.mock.calls.map(([v]) => v)).toEqual([true, false])
      await t.stop() // not active: no-op
      await t.release()
    })

    it('writes the audio to a WAV file when configured', async () => {
      const stream = createStream()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, audioStream: stream, fs: {} as any },
        { audioOutputPath: '/tmp/out.wav' },
      )
      await t.start()
      stream.simulateDataChunk(new Uint8Array(3200))
      await settle()
      expect(mockWavFileWriter.initialize).toHaveBeenCalled()
      expect(mockWavFileWriter.appendAudioData).toHaveBeenCalled()
      await t.stop()
      expect(mockWavFileWriter.finalize).toHaveBeenCalled()
      await t.release()
    })
  })

  describe('segmentation with a VAD', () => {
    const vadOptions = { threshold: 0.5, minSpeechDurationMs: 100, minSilenceDurationMs: 200, speechPadMs: 30, maxSpeechDurationS: 30 }

    it('finalizes a slice per speech segment with its exact audio and position', async () => {
      const stream = createStream()
      const { context, calls } = createWhisper()
      const onTranscribe = jest.fn()
      const onVad = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: context, vadContext: createFakeVad(), audioStream: stream },
        { vadOptions, realtimeProcessingPauseMs: 1e9 },
        { onTranscribe, onVad },
      )
      await t.start()
      const audio = makeAudio([[500, false], [1000, true], [600, false]])
      await feed(stream, audio)

      const [final] = finalsOf(onTranscribe)
      expect(final!.endReason).toBe('pause')
      // Speech at 8000-24000, padded by 480, on 512-sample hops
      expect(Math.abs(final!.startSample! - (8000 - 480))).toBeLessThan(HOP)
      expect(Math.abs(final!.endSample! - (24000 + 480))).toBeLessThan(HOP)
      const finalCall = calls[calls.length - 1]!
      expect(Array.from(finalCall.samples)).toEqual(Array.from(audio.slice(final!.startSample!, final!.endSample!)))
      expect(onVad.mock.calls.map(([e]) => e.type)).toEqual(
        expect.arrayContaining(['speech_start', 'speech_end']),
      )
      await t.release()
    })

    it('splits long speech at an earlier pause, every sample in exactly one final', async () => {
      const stream = createStream()
      const { context, calls } = createWhisper()
      const onTranscribe = jest.fn()
      const split = { ...vadOptions, maxSpeechDurationS: 1, minSilenceDurationMs: 300 }
      const t = new RealtimeTranscriber(
        { whisperContext: context, vadContext: createFakeVad(), audioStream: stream },
        { vadOptions: split, realtimeProcessingPauseMs: 1e9 },
        { onTranscribe },
      )
      await t.start()
      const audio = makeAudio([[300, false], [600, true], [200, false], [900, true], [120, false], [500, true], [400, false]])
      await feed(stream, audio, 10)

      // The same segmentation offline, with the energy StreamVad uses
      const segmenter = new StreamingVadSegmenter(split, HOP, SR, {
        frameEnergy: (from, to) => audio.slice(from, to).reduce((s, v) => s + (v / 32768) ** 2, 0) / (to - from),
      })
      const probs: number[] = []
      for (let i = 0; i < audio.length; i += HOP) {
        const hop = audio.subarray(i, i + HOP)
        probs.push(hop.reduce((s, v) => s + Math.abs(v), 0) / hop.length > 1000 ? 0.9 : 0.05)
      }
      const lastFull = Math.floor(audio.length / HOP)
      const bounds = [...segmenter.push(probs.slice(0, lastFull)), ...segmenter.flush(audio.length, probs.slice(lastFull))]
      const expected: number[][] = []
      for (let i = 0; i < bounds.length; i += 2) {
        expected.push(Array.from(audio.slice(bounds[i]!.sample, bounds[i + 1]!.sample)))
      }
      expect(bounds.some((b) => b.type === 'end' && b.reason === 'max_duration')).toBe(true)

      const finals = finalsOf(onTranscribe)
      const finalAudio = finals.map((e) => {
        const index = transcribeEvents(onTranscribe).indexOf(e)
        return Array.from(calls[index]!.samples)
      })
      expect(finalAudio).toEqual(expected)
      expect(finals.map((e) => e.endReason)).toContain('max_duration')
      await t.release()
    })
  })

  describe('with a segmenter VAD', () => {
    // A segmenter the test drives; `positions` adds the optional sample positions
    const createManualSegmenter = () => {
      const handlers: any = {}
      const segmenter = {
        processAudio: jest.fn(),
        onSpeechStart: (cb: any) => { handlers.start = cb },
        onSpeechContinue: (cb: any) => { handlers.continue = cb },
        onSpeechEnd: (cb: any) => { handlers.end = cb },
        onError: jest.fn(),
        updateOptions: jest.fn(),
        flush: jest.fn(async () => undefined),
        reset: jest.fn(async () => undefined),
        release: jest.fn(async () => undefined),
      }
      return { segmenter, handlers }
    }

    it('cuts at the reported end when the segmenter gives positions', async () => {
      const { segmenter, handlers } = createManualSegmenter()
      const onTranscribe = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, vadContext: segmenter, audioStream: createStream() },
        { realtimeProcessingPauseMs: 1e9 },
        { onTranscribe },
      )
      await t.start()
      handlers.start(0.9, bytesOf(makeAudio([[1000, true]])), { startSample: 0 })
      handlers.end(0.9, { endSample: 8000, reason: 'max_duration' })
      // The next segment re-sends from 7000: only what was not final yet is kept
      handlers.start(0.9, bytesOf(makeAudio([[500, true]])), { startSample: 7000 })
      handlers.end(0.9, { endSample: 15000, reason: 'pause' })
      await settle()
      expect(finalsOf(onTranscribe).map((e) => [e.endReason, e.startSample, e.endSample])).toEqual([
        ['max_duration', 0, 8000],
        ['pause', 8000, 15000],
      ])
      await t.release()
      expect(segmenter.release).toHaveBeenCalled()
    })

    it('follows a segmenter without positions, bounding slices by maxSpeechDurationS', async () => {
      const { segmenter, handlers } = createManualSegmenter()
      const onTranscribe = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, vadContext: segmenter, audioStream: createStream() },
        { vadOptions: { maxSpeechDurationS: 1 }, realtimeProcessingPauseMs: 1e9 },
        { onTranscribe },
      )
      await t.start()
      handlers.start(0.9, bytesOf(makeAudio([[500, true]])))
      for (let i = 0; i < 5; i += 1) handlers.continue(0.9, bytesOf(makeAudio([[250, true]])))
      handlers.end(0.9)
      await settle()
      const finals = finalsOf(onTranscribe)
      expect(finals.map((e) => e.endReason)).toEqual(['max_duration', undefined])
      expect(finals.map((e) => e.endSample! - e.startSample!)).toEqual([SR, 0.75 * SR])
      await t.release()
    })
  })

  describe('without a VAD', () => {
    it('cuts continuous audio into slices of at most maxSpeechDurationS', async () => {
      const stream = createStream()
      const onTranscribe = jest.fn()
      const onVad = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, audioStream: stream },
        { vadOptions: { maxSpeechDurationS: 1 }, realtimeProcessingPauseMs: 1e9 },
        { onTranscribe, onVad },
      )
      await t.start()
      await feed(stream, makeAudio([[3500, true]]))
      await t.stop()

      const finals = finalsOf(onTranscribe)
      expect(finals.length).toBeGreaterThanOrEqual(4)
      finals.forEach((e) => expect(e.endSample! - e.startSample!).toBeLessThanOrEqual(SR))
      // Contiguous, all audio covered
      finals.forEach((e, i) => {
        if (i > 0) expect(e.startSample).toBe(finals[i - 1]!.endSample)
      })
      expect(finals[finals.length - 1]).toMatchObject({ endReason: 'end_of_stream', endSample: 3.5 * SR })
      expect(finals.slice(0, -1).every((e) => e.endReason === 'max_duration')).toBe(true)
      expect(onVad).not.toHaveBeenCalled()
      await t.release()
    })

    it('keeps the maximum finite', async () => {
      const logger = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, audioStream: createStream() },
        { vadOptions: { maxSpeechDurationS: 0 }, logger },
      )
      expect(logger).toHaveBeenCalledWith(expect.stringContaining('maxSpeechDurationS must be finite'))
      await t.release()
    })

    it('does not take an infinite maximum, at construction or later', async () => {
      const logger = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, audioStream: createStream() },
        { vadOptions: { maxSpeechDurationS: Infinity }, logger },
      )
      expect(logger).toHaveBeenCalledTimes(1)
      t.updateVadOptions({ maxSpeechDurationS: Infinity })
      expect(logger).toHaveBeenCalledTimes(2)
      expect(logger.mock.calls.every(([m]) => /maxSpeechDurationS must be finite/.test(m))).toBe(true)
      await t.release()
    })

    it('says that audioSliceSec is gone', async () => {
      const logger = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, audioStream: createStream() },
        { audioSliceSec: 10, logger } as any,
      )
      expect(logger).toHaveBeenCalledWith(expect.stringContaining('audioSliceSec'))
      await t.release()
    })

    it('finalizes on nextSlice() and continues without losing or repeating audio', async () => {
      const stream = createStream()
      const onTranscribe = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, audioStream: stream },
        { realtimeProcessingPauseMs: 1e9 },
        { onTranscribe },
      )
      await t.start()
      await feed(stream, makeAudio([[1000, true]]))
      await t.nextSlice()
      await feed(stream, makeAudio([[500, true]]))
      await t.stop()
      const finals = finalsOf(onTranscribe)
      // nextSlice() cuts at the audio handed out so far (StreamVad holds back the last hop)
      const cut = finals[0]!.endSample!
      expect(cut).toBeGreaterThan(16000 - 2 * HOP)
      expect(finals.map((e) => [e.endReason, e.startSample, e.endSample])).toEqual([
        ['manual', 0, cut],
        ['end_of_stream', cut, 24000],
      ])
      await t.release()
    })
  })

  describe('drafts and scheduling', () => {
    it('throttles drafts by audio time, not wall-clock time', async () => {
      const stream = createStream()
      const onBeginTranscribe = jest.fn().mockResolvedValue(true)
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, audioStream: stream },
        { realtimeProcessingPauseMs: 500, initRealtimeAfterMs: 200 },
        { onBeginTranscribe },
      )
      await t.start()
      await feed(stream, makeAudio([[3000, true]]))
      const durations = onBeginTranscribe.mock.calls
        .filter(([info]) => !info.isFinal)
        .map(([info]) => info.duration)
      // Every 500 ms of audio from 200 ms on (in hops as the audio is handed out)
      expect(durations).toHaveLength(6)
      expect(durations[0]).toBeGreaterThanOrEqual(200)
      durations.slice(1).forEach((d, i) => {
        expect(d - durations[i]!).toBeGreaterThanOrEqual(500)
        expect(d - durations[i]!).toBeLessThan(500 + 100 + 32)
      })
      await t.release()
    })

    it('applies draftTranscribeOptions to drafts only', async () => {
      const stream = createStream()
      const { context, calls } = createWhisper()
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        {
          transcribeOptions: { language: 'en', maxThreads: 2 },
          draftTranscribeOptions: { maxTokens: 16, singleSegment: true, audioCtx: 'auto' },
          realtimeProcessingPauseMs: 1e9,
        },
      )
      await t.start()
      await feed(stream, makeAudio([[500, true]]))
      await t.stop()
      const [draft, final] = calls
      expect(draft!.options).toMatchObject({ language: 'en', maxThreads: 2, maxTokens: 16, singleSegment: true, audioCtx: 'auto' })
      expect(final!.options).toMatchObject({ language: 'en', maxThreads: 2 })
      expect(final!.options).not.toHaveProperty('maxTokens')
      await t.release()
    })

    it('runs a final before a waiting draft and aborts a running draft for it', async () => {
      const stream = createStream()
      const onTranscribe = jest.fn()
      const stops: jest.Mock[] = []
      let resolveDraft: (r: TranscribeResult) => void = () => {}
      const context = {
        transcribeData: jest.fn((_data: ArrayBuffer, options: any) => {
          if (context.transcribeData.mock.calls.length === 1) {
            // The first draft hangs until it is stopped
            const stop = jest.fn(async () => resolveDraft({ isAborted: true, language: 'en', result: 'partial', segments: [] }))
            stops.push(stop)
            return { stop, promise: new Promise<TranscribeResult>((resolve) => { resolveDraft = resolve }) }
          }
          return {
            stop: jest.fn(),
            promise: Promise.resolve({ isAborted: false, language: 'en', result: options.prompt ? 'x' : 'final', segments: [] }),
          }
        }),
      }
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { realtimeProcessingPauseMs: 1e9, promptPreviousSlices: false },
        { onTranscribe },
      )
      await t.start()
      await feed(stream, makeAudio([[500, true]]))
      await t.nextSlice()
      await settle()
      expect(stops[0]).toHaveBeenCalled()
      // The aborted draft reports nothing; the final does
      expect(transcribeEvents(onTranscribe).map((e) => [e.isFinal, e.data?.result])).toEqual([[true, 'final']])
      await t.release()
    })

    it('keeps at most one waiting draft while transcription is slow', async () => {
      let release: () => void = () => {}
      const blocked = new Promise<void>((resolve) => { release = resolve })
      const sizes: number[] = []
      const slow = {
        transcribeData: jest.fn((data: ArrayBuffer) => {
          sizes.push(data.byteLength)
          return { stop: jest.fn(), promise: blocked.then(() => ({ isAborted: false, language: 'en', result: 'text', segments: [] })) }
        }),
      }
      const stream = createStream()
      const t = new RealtimeTranscriber(
        { whisperContext: slow, audioStream: stream },
        { realtimeProcessingPauseMs: 1, initRealtimeAfterMs: 1 },
      )
      await t.start()
      await feed(stream, makeAudio([[3000, true]]))
      expect((t as any).transcriptionQueue.length).toBeLessThanOrEqual(1)
      expect(sizes[0]).toBeLessThan(3 * SR * 2)
      release()
      await t.stop()
      await t.release()
    })

    it('tells onBeginTranscribe whether a pass is final, so drafts can be skipped', async () => {
      const stream = createStream()
      const { context, calls } = createWhisper()
      const onBeginTranscribe = jest.fn(async ({ isFinal }) => isFinal)
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { realtimeProcessingPauseMs: 100 },
        { onBeginTranscribe },
      )
      await t.start()
      await feed(stream, makeAudio([[1000, true]]))
      await t.stop()
      expect(onBeginTranscribe.mock.calls.some(([info]) => !info.isFinal)).toBe(true)
      expect(calls).toHaveLength(1) // the final only
      await t.release()
    })

    it('transcribes open speech on stop() and drops pending drafts', async () => {
      const stream = createStream()
      const onTranscribe = jest.fn()
      const t = new RealtimeTranscriber(
        { whisperContext: createWhisper().context, vadContext: createFakeVad(), audioStream: stream },
        { realtimeProcessingPauseMs: 1e9 },
        { onTranscribe },
      )
      await t.start()
      await feed(stream, makeAudio([[300, false], [800, true]]))
      await t.stop()
      expect(finalsOf(onTranscribe).map((e) => e.endReason)).toEqual(['end_of_stream'])
      await t.release()
    })

    it('reports transcription errors', async () => {
      const stream = createStream()
      const onError = jest.fn()
      const failing = {
        transcribeData: jest.fn(() => ({ stop: jest.fn(), promise: Promise.reject(new Error('boom')) })),
      }
      const t = new RealtimeTranscriber({ whisperContext: failing, audioStream: stream }, {}, { onError })
      await t.start()
      await feed(stream, makeAudio([[500, true]]))
      expect(onError).toHaveBeenCalledWith(expect.stringContaining('boom'))
      await t.release()
    })
  })

  it('transcribes every final of a backlog longer than maxSlicesInMemory', async () => {
    let release: () => void = () => {}
    const gate = new Promise<void>((resolve) => { release = resolve })
    const whisper = {
      transcribeData: jest.fn(() => ({
        stop: jest.fn(),
        promise: gate.then(() => ({ isAborted: false, language: 'en', result: 'x', segments: [] })),
      })),
    }
    const stream = createStream()
    const onTranscribe = jest.fn()
    const t = new RealtimeTranscriber(
      { whisperContext: whisper, audioStream: stream },
      { maxSlicesInMemory: 1, initRealtimeAfterMs: 1e9 },
      { onTranscribe },
    )
    await t.start()
    for (let i = 0; i < 5; i += 1) {
      // eslint-disable-next-line no-await-in-loop
      await feed(stream, makeAudio([[100, true]]))
      // eslint-disable-next-line no-await-in-loop
      await t.nextSlice()
    }
    release()
    await settle()
    expect(finalsOf(onTranscribe)).toHaveLength(5)
    await t.release()
  })

  describe('transcribe options', () => {
    it('passes audioCtx through to Whisper', async () => {
      const stream = createStream()
      const { context, calls } = createWhisper()
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { transcribeOptions: { maxThreads: 2, audioCtx: 'auto' } },
      )
      await t.start()
      await feed(stream, makeAudio([[500, true]]))
      expect(calls[0]!.options).toMatchObject({ maxThreads: 2, audioCtx: 'auto' })
      await t.release()
    })

    it('gives Parakeet only its own options', async () => {
      const stream = createStream()
      const parakeet = {
        transcribeData: jest.fn((_data: ArrayBuffer, _options: ParakeetTranscribeOptions) => ({
          stop: jest.fn().mockResolvedValue(undefined),
          promise: Promise.resolve({ isAborted: false, language: '', result: 'Parakeet', segments: [] }),
        })),
      }
      const t = new RealtimeTranscriber(
        { parakeetContext: parakeet, audioStream: stream },
        {
          transcribeOptions: { maxThreads: 2, audioCtx: 'auto' },
          initialPrompt: 'ignored',
          languageLock: true,
          splitOverlapMs: 400,
        },
      )
      await t.start()
      await feed(stream, makeAudio([[500, true]]))
      expect(parakeet.transcribeData.mock.calls[0]![1]).toEqual({ maxThreads: 2 })
      await t.release()
    })

    it('chains previous results into the Whisper prompt', async () => {
      const stream = createStream()
      const { context, calls } = createWhisper((_c, i) => ({ result: `r${i}` }))
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { initialPrompt: 'Init', realtimeProcessingPauseMs: 1e9, initRealtimeAfterMs: 1e9 },
      )
      await t.start()
      await feed(stream, makeAudio([[300, true]]))
      await t.nextSlice()
      await settle()
      await feed(stream, makeAudio([[300, true]]))
      await t.nextSlice()
      await settle()
      expect(calls.map((c) => c.options.prompt)).toEqual(['Init', 'Init r0'])
      await t.release()
    })
  })

  describe('quality', () => {
    it('spots repetition loops', () => {
      expect(isRepetitive('And then we have a new data set. '.repeat(4))).toBe(true)
      expect(isRepetitive('Lexical borrowing is a type of linguistic borrowing, which is basically reproducing patterns.')).toBe(false)
      expect(isRepetitive('我們有一個新的資料集'.repeat(3))).toBe(true)
      expect(isRepetitive('short. short. short.')).toBe(false)
    })

    it('keeps repetition loops out of the prompt', async () => {
      const stream = createStream()
      const loop = 'And then we have a new data set. '.repeat(4).trim()
      const { context, calls } = createWhisper((_c, i) => ({ result: i === 0 ? loop : `r${i}` }))
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { realtimeProcessingPauseMs: 1e9, initRealtimeAfterMs: 1e9 },
      )
      await t.start()
      for (let i = 0; i < 3; i += 1) {
        // eslint-disable-next-line no-await-in-loop
        await feed(stream, makeAudio([[300, true]]))
        // eslint-disable-next-line no-await-in-loop
        await t.nextSlice()
        // eslint-disable-next-line no-await-in-loop
        await settle()
      }
      expect(calls.map((c) => c.options.prompt)).toEqual([undefined, undefined, 'r1'])
      await t.release()
    })

    it('chains no previous text across an in-speech split, even with a backlog', async () => {
      let release: () => void = () => {}
      const gate = new Promise<void>((resolve) => { release = resolve })
      const prompts: Array<string | undefined> = []
      const whisper = {
        transcribeData: jest.fn((_data: ArrayBuffer, options: any) => {
          const i = prompts.push(options.prompt) - 1
          return {
            stop: jest.fn(),
            promise: gate.then(() => ({ isAborted: false, language: 'en', result: `S${i}. tail${i}`, segments: [] })),
          }
        }),
      }
      const stream = createStream()
      const t = new RealtimeTranscriber(
        { whisperContext: whisper, audioStream: stream },
        { vadOptions: { maxSpeechDurationS: 1 }, realtimeProcessingPauseMs: 1e9, initRealtimeAfterMs: 1e9 },
      )
      await t.start()
      // Every slice is cut inside speech, and all are cut before any is transcribed
      await feed(stream, makeAudio([[3500, true]]))
      release()
      await t.stop()
      expect(prompts.length).toBeGreaterThanOrEqual(4)
      expect(prompts.every((p) => p === undefined)).toBe(true)
      await t.release()
    })

    it('drops segments below minAvgLogprob', async () => {
      const stream = createStream()
      const onTranscribe = jest.fn()
      const { context } = createWhisper(() => ({
        result: ' real speech gibberish',
        segments: [
          { text: ' real speech', t0: 0, t1: 50, avgLogprob: -0.3 },
          { text: ' gibberish', t0: 50, t1: 100, avgLogprob: -2.4 },
        ],
      }))
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { minAvgLogprob: -1.5, realtimeProcessingPauseMs: 1e9, initRealtimeAfterMs: 1e9 },
        { onTranscribe },
      )
      await t.start()
      await feed(stream, makeAudio([[300, true]]))
      await t.stop()
      const [final] = finalsOf(onTranscribe)
      expect(final!.data!.result).toBe('real speech')
      expect(final!.data!.segments).toHaveLength(1)
      await t.release()
    })
  })

  describe('language lock', () => {
    it('passes the language explicitly once it was detected confidently', async () => {
      const stream = createStream()
      const { context, calls } = createWhisper(() => ({ result: 'x', language: 'ja', languageProb: 0.95 }))
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { languageLock: { lockAfter: 3 }, realtimeProcessingPauseMs: 1e9, initRealtimeAfterMs: 1e9 },
      )
      await t.start()
      for (let i = 0; i < 4; i += 1) {
        // eslint-disable-next-line no-await-in-loop
        await feed(stream, makeAudio([[300, true]]))
        // eslint-disable-next-line no-await-in-loop
        await t.nextSlice()
        // eslint-disable-next-line no-await-in-loop
        await settle()
      }
      expect(calls.map((c) => c.options.language)).toEqual(['auto', 'auto', 'auto', 'ja'])
      expect(t.lockedLanguage).toBe('ja')
      await t.release()
    })
  })

  describe('overlap around in-speech splits', () => {
    // Audio whose samples encode their own position (value = position / 2), and
    // a Whisper stand-in that "hears" a word every 100 ms of the stream: word k
    // starts at sample k * 1600 and is reported when its start is in the audio.
    const positionAudio = (samples: number) => {
      const audio = new Int16Array(samples)
      for (let i = 0; i < samples; i += 1) audio[i] = Math.floor(i / 2)
      return audio
    }
    const wordWhisper = () =>
      createWhisper(({ samples }) => {
        const first = samples[0]! * 2
        const tokens = []
        for (let k = Math.ceil(first / 1600); k * 1600 < first + samples.length; k += 1) {
          tokens.push({ id: k, text: ` w${k}`, p: 0.9, plog: -0.1, t0: Math.round((k * 1600 - first) / 160), t1: -1, tDtw: -1 })
        }
        const text = tokens.map((tk) => tk.text).join('')
        return { result: text, segments: [{ text, t0: 0, t1: 0, tokens }] }
      })

    it('keeps every word exactly once across slices', async () => {
      const stream = createStream()
      const onTranscribe = jest.fn()
      const { context, calls } = wordWhisper()
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { vadOptions: { maxSpeechDurationS: 1.5 }, splitOverlapMs: 400, realtimeProcessingPauseMs: 1e9, initRealtimeAfterMs: 1e9 },
        { onTranscribe },
      )
      await t.start()
      const total = 4 * SR
      await feed(stream, positionAudio(total))
      await t.stop()

      const finals = finalsOf(onTranscribe)
      expect(finals.length).toBeGreaterThanOrEqual(3)
      // Slices around in-speech splits overlap in audio
      expect(finals[1]!.startSample!).toBeLessThan(finals[0]!.endSample!)
      // Final requests ask for tokens with times
      expect(calls.filter((c) => c.options.includeTokens && c.options.tokenTimestamps).length).toBeGreaterThan(0)
      const words = finals.map((e) => e.data!.result).join(' ').split(/\s+/).filter(Boolean)
      const expected = Array.from({ length: total / 1600 }, (_, k) => `w${k}`)
      expect(words).toEqual(expected)
      await t.release()
    })

    it('gives the right-hand overlap even when one chunk spans the split', async () => {
      const stream = createStream()
      const onTranscribe = jest.fn()
      const { context, calls } = wordWhisper()
      const t = new RealtimeTranscriber(
        { whisperContext: context, audioStream: stream },
        { vadOptions: { maxSpeechDurationS: 2 }, splitOverlapMs: 400, realtimeProcessingPauseMs: 1e9, initRealtimeAfterMs: 1e9 },
        { onTranscribe },
      )
      await t.start()
      const total = 4 * SR
      // The whole audio in one chunk: the split falls inside it
      await feed(stream, positionAudio(total), 4000)
      await t.stop()

      const finals = finalsOf(onTranscribe)
      const first = finals[0]!
      expect(first.endReason).toBe('max_duration')
      // The next slice starts 400 ms before the cut (left overlap); the first
      // final's audio reaches 400 ms past it
      const cut = finals[1]!.startSample! + 0.4 * SR
      const firstCall = calls.find((c) => c.samples[0] === 0)!
      expect(firstCall.samples.length).toBe(cut + 0.4 * SR)
      const words = finals.map((e) => e.data!.result).join(' ').split(/\s+/).filter(Boolean)
      expect(words).toEqual(Array.from({ length: total / 1600 }, (_, k) => `w${k}`))
      await t.release()
    })
  })

  describe('results and statistics', () => {
    const runFinalSlices = async (t: RealtimeTranscriber, stream: JestAudioStreamAdapter, count: number) => {
      for (let i = 0; i < count; i += 1) {
        // eslint-disable-next-line no-await-in-loop
        await feed(stream, makeAudio([[100, true]]))
        // eslint-disable-next-line no-await-in-loop
        await t.nextSlice()
        // eslint-disable-next-line no-await-in-loop
        await settle()
      }
    }
    const harness = (options: Record<string, unknown> = {}) => {
      const stream = createStream()
      const whisper = createWhisper((_c, i) => ({ result: `slice ${i}` }))
      const t = new RealtimeTranscriber(
        { whisperContext: whisper.context, audioStream: stream },
        { initRealtimeAfterMs: 1e9, maxSlicesInMemory: 3, ...options },
      )
      const lastPrompt = () => whisper.calls[whisper.calls.length - 1]?.options.prompt
      return { t, stream, lastPrompt }
    }

    it('retains only the newest maxResultsInMemory results', async () => {
      const { t, stream } = harness({ maxResultsInMemory: 5 })
      await t.start()
      await runFinalSlices(t, stream, 20)
      const results = t.getTranscriptionResults()
      expect(results.map((r) => r.slice.index)).toEqual([15, 16, 17, 18, 19])
      expect(results.map((r) => r.transcribeEvent.data?.result)).toEqual([15, 16, 17, 18, 19].map((i) => `slice ${i}`))
      await t.release()
    })

    it('bounds the previous-slice prompt to maxPromptSlices', async () => {
      const { t, stream, lastPrompt } = harness({ initialPrompt: 'Init', maxPromptSlices: 2 })
      await t.start()
      await runFinalSlices(t, stream, 10)
      expect(lastPrompt()).toBe('Init slice 7 slice 8')
      await t.release()
    })

    it('reports statistics', async () => {
      const onStatsUpdate = jest.fn()
      const { t, stream } = harness()
      t.updateCallbacks({ onStatsUpdate })
      await t.start()
      await runFinalSlices(t, stream, 2)
      expect(onStatsUpdate).toHaveBeenCalled()
      expect(t.getStatistics()).toMatchObject({ isActive: true, vadEnabled: false, sliceStats: expect.any(Object) })
      await t.release()
    })
  })

  describe('errors and cleanup', () => {
    it('reports audio stream errors', async () => {
      const stream = createStream()
      const onError = jest.fn()
      const t = new RealtimeTranscriber({ whisperContext: createWhisper().context, audioStream: stream }, {}, { onError })
      stream.simulateError('mic gone')
      expect(onError).toHaveBeenCalledWith('mic gone')
      await t.release()
    })

    it('stays inactive when the audio stream fails to start', async () => {
      const stream = createStream()
      jest.spyOn(stream, 'start').mockRejectedValue(new Error('no mic'))
      const t = new RealtimeTranscriber({ whisperContext: createWhisper().context, audioStream: stream })
      await expect(t.start()).rejects.toThrow('no mic')
      expect(t.getStatistics().isActive).toBe(false)
      await t.release()
    })

    it('releases the audio stream and its VAD stream, not the VAD context', async () => {
      const stream = createStream()
      const release = jest.spyOn(stream, 'release')
      const vad = createFakeVad()
      const streamRelease = jest.fn(async () => undefined)
      const createStreamOrig = vad.createStream
      vad.createStream = async () => ({ ...(await createStreamOrig()), release: streamRelease })
      const t = new RealtimeTranscriber({ whisperContext: createWhisper().context, vadContext: vad, audioStream: stream })
      await t.start()
      await feed(stream, makeAudio([[200, true]]))
      await t.release()
      expect(release).toHaveBeenCalled()
      expect(streamRelease).toHaveBeenCalled()
    })
  })
})
