import type { VadOptions } from '../index'
import type { RealtimeVadContextLike } from './types'
import { VAD_PRESETS } from './types'
import { StreamingVadSegmenter } from './StreamingVadSegmenter'

/** The part of `WhisperVadStream` StreamVad uses */
export type VadStreamLike = {
  push(audio: Uint8Array): Promise<number[]>
  flush(): Promise<number[]>
  reset(): Promise<void>
  release(): Promise<void>
}

/** The part of `WhisperVadContext` StreamVad uses */
export type StreamVadContextLike = {
  hopSamples: number
  sampleRate: number
  createStream(): Promise<VadStreamLike>
}

export type StreamVadOptions = {
  vadOptions?: VadOptions
  vadPreset?: keyof typeof VAD_PRESETS
  logger?: (message: string) => void
}

const BYTES_PER_SAMPLE = 2

/** PCM16 audio addressed by absolute sample index, dropped from the front once no longer needed. */
class PcmQueue {
  private chunks: Uint8Array[] = []

  /** Absolute sample index of chunks[0][0] */
  private base = 0

  private total = 0

  get end(): number {
    return this.total
  }

  append(data: Uint8Array): void {
    if (data.byteLength === 0) return
    this.chunks.push(data)
    this.total += data.byteLength / BYTES_PER_SAMPLE
  }

  /** Samples [from, to) that are still held */
  read(from: number, to: number): Uint8Array {
    const start = Math.max(from, this.base)
    const end = Math.min(to, this.total)
    if (end <= start) return new Uint8Array(0)
    const out = new Uint8Array((end - start) * BYTES_PER_SAMPLE)
    let chunkStart = this.base
    let offset = 0
    // `some` stops once the chunk holding `end` is copied
    this.chunks.some((chunk) => {
      const chunkEnd = chunkStart + chunk.byteLength / BYTES_PER_SAMPLE
      if (chunkEnd > start && chunkStart < end) {
        const a = Math.max(start, chunkStart) - chunkStart
        const b = Math.min(end, chunkEnd) - chunkStart
        out.set(chunk.subarray(a * BYTES_PER_SAMPLE, b * BYTES_PER_SAMPLE), offset)
        offset += (b - a) * BYTES_PER_SAMPLE
      }
      chunkStart = chunkEnd
      return chunkEnd >= end
    })
    return out
  }

  /** Drop whole chunks that end at or before `sample` */
  dropBefore(sample: number): void {
    while (this.chunks.length > 0) {
      const first = this.chunks[0]!
      const firstEnd = this.base + first.byteLength / BYTES_PER_SAMPLE
      if (firstEnd > sample) break
      this.chunks.shift()
      this.base = firstEnd
    }
  }

  clear(): void {
    this.chunks = []
    this.base = 0
    this.total = 0
  }
}

/**
 * Stateful VAD for `RealtimeTranscriber`, on a `WhisperVadStream`.
 *
 * Every incoming chunk is pushed to the stream once; the model state carries
 * over, so there is one exact speech probability per hop (32 ms for Silero)
 * instead of re-running a window of recent audio. `StreamingVadSegmenter`
 * turns the probabilities into segments with `VadOptions`, and the speech
 * events carry the segment's audio:
 * - `onSpeechStart`: confirmed speech, with the audio from its padded start;
 * - `onSpeechContinue`: further audio of the segment, each sample once;
 * - `onSpeechEnd`: after the audio up to the padded end.
 *
 * Positions are counted in samples of the pushed audio, not wall-clock time.
 * Input: 16 kHz mono PCM16.
 */
export class StreamVad implements RealtimeVadContextLike {
  private readonly vadContext: StreamVadContextLike

  private readonly segmenter: StreamingVadSegmenter

  private readonly audio = new PcmQueue()

  private readonly logger: (message: string) => void

  private vadOptions: VadOptions

  private stream: VadStreamLike | null = null

  /** The stream received a flush and must be reset before more audio */
  private streamEnded = false

  private queue: Promise<void> = Promise.resolve()

  /** End of the audio already handed to the speech callbacks */
  private emittedUpTo = 0

  private speechOpen = false

  /** onSpeechStart is still due for the open segment */
  private startPending = false

  private lastProb = 0

  private speechStartCallback: ((confidence: number, data: Uint8Array) => void) | null = null

  private speechContinueCallback: ((confidence: number, data: Uint8Array) => void) | null = null

  private speechEndCallback: ((confidence: number) => void) | null = null

  private errorCallback: ((error: string) => void) | null = null

  constructor(vadContext: StreamVadContextLike, options: StreamVadOptions = {}) {
    this.vadContext = vadContext
    this.logger = options.logger || (() => {})
    const preset = options.vadPreset ? VAD_PRESETS[options.vadPreset] : VAD_PRESETS.default
    this.vadOptions = { ...preset, ...options.vadOptions }
    this.segmenter = new StreamingVadSegmenter(
      this.vadOptions,
      vadContext.hopSamples,
      vadContext.sampleRate,
    )
  }

  onSpeechStart(callback: (confidence: number, data: Uint8Array) => void): void {
    this.speechStartCallback = callback
  }

  onSpeechContinue(callback: (confidence: number, data: Uint8Array) => void): void {
    this.speechContinueCallback = callback
  }

  onSpeechEnd(callback: (confidence: number) => void): void {
    this.speechEndCallback = callback
  }

  onError(callback: (error: string) => void): void {
    this.errorCallback = callback
  }

  updateOptions(options: Partial<VadOptions>): void {
    this.vadOptions = { ...this.vadOptions, ...options }
    this.segmenter.setOptions(this.vadOptions)
  }

  processAudio(data: Uint8Array): void {
    // Copy: the caller may reuse its buffer
    const chunk = data.slice()
    this.enqueue(() => this.handleAudio(chunk))
  }

  /** End of audio: emits the remaining probabilities and closes open speech. */
  async flush(): Promise<void> {
    this.enqueue(() => this.handleFlush())
    await this.queue
  }

  /** Waits for queued audio, then starts over (model state included). */
  async reset(): Promise<void> {
    this.enqueue(async () => {
      if (this.stream) await this.stream.reset()
      this.streamEnded = false
      this.clearState()
    })
    await this.queue
  }

  /** Frees the native stream. */
  async release(): Promise<void> {
    this.enqueue(async () => {
      const { stream } = this
      this.stream = null
      this.clearState()
      if (stream) await stream.release()
    })
    await this.queue
  }

  private enqueue(task: () => Promise<void>): void {
    this.queue = this.queue.then(task).catch((error: any) => {
      this.log(`VAD error: ${error?.message || error}`)
      this.errorCallback?.(`VAD processing error: ${error?.message || error}`)
    })
  }

  private clearState(): void {
    this.segmenter.reset()
    this.audio.clear()
    this.emittedUpTo = 0
    this.speechOpen = false
    this.startPending = false
    this.lastProb = 0
  }

  private async ensureStream(): Promise<VadStreamLike> {
    if (!this.stream) {
      this.stream = await this.vadContext.createStream()
    } else if (this.streamEnded) {
      // Audio after a flush is a new stream
      await this.stream.reset()
      this.clearState()
    }
    this.streamEnded = false
    return this.stream
  }

  private async handleAudio(chunk: Uint8Array): Promise<void> {
    const stream = await this.ensureStream()
    this.audio.append(chunk)
    const probs = await stream.push(chunk)
    this.consume(probs)
  }

  private async handleFlush(): Promise<void> {
    if (!this.stream || this.streamEnded) return
    const probs = await this.stream.flush()
    this.streamEnded = true
    this.consume(probs)
    this.apply(this.segmenter.flush(this.audio.end))
    this.audio.dropBefore(this.audio.end)
  }

  private consume(probs: number[]): void {
    if (probs.length > 0) this.lastProb = probs[probs.length - 1]!
    this.apply(this.segmenter.push(probs))
    if (this.speechOpen) {
      this.emitUntil(this.segmenter.speechCertainUntil)
    }
    this.audio.dropBefore(Math.max(this.emittedUpTo, this.segmenter.retainFrom))
  }

  private apply(events: ReturnType<StreamingVadSegmenter['push']>): void {
    events.forEach((event) => {
      if (event.type === 'start') {
        this.speechOpen = true
        this.startPending = true
        this.lastProb = event.prob
        // A padded start never reaches back into audio already handed out
        this.emittedUpTo = Math.max(this.emittedUpTo, event.sample)
        this.log(`speech start at ${event.sample}`)
      } else if (this.speechOpen) {
        this.emitUntil(event.sample)
        this.speechOpen = false
        this.startPending = false
        this.log(`speech end at ${Math.min(event.sample, this.audio.end)}`)
        this.speechEndCallback?.(1 - event.prob)
      }
    })
  }

  private emitUntil(sample: number): void {
    const to = Math.min(sample, this.audio.end)
    const data = this.audio.read(this.emittedUpTo, to)
    if (to > this.emittedUpTo) this.emittedUpTo = to
    if (this.startPending) {
      this.startPending = false
      this.speechStartCallback?.(this.lastProb, data)
    } else if (data.byteLength > 0) {
      this.speechContinueCallback?.(this.lastProb, data)
    }
  }

  private log(message: string) {
    this.logger(`[StreamVad] ${message}`)
  }
}
