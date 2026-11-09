import type { VadOptions } from '../index'

export type VadSegmenterEvent =
  | {
      /** A speech segment is confirmed. `sample` is its padded start. */
      type: 'start'
      sample: number
      /** Probability of the frame that triggered the segment */
      prob: number
    }
  | {
      /** The current speech segment ended. `sample` is its padded end (exclusive). */
      type: 'end'
      sample: number
      /** Probability of the frame that decided the end */
      prob: number
    }

/**
 * Incremental version of silero-vad's `get_speech_timestamps_from_probs`, the
 * segmentation vad.cpp and `WhisperVadContext.detectSpeech*` use offline.
 *
 * It consumes speech probabilities as a `WhisperVadStream` produces them (one
 * per hop) and reports segments in absolute samples as soon as they are
 * certain:
 * - hysteresis: speech starts at `prob >= threshold` and only a run of
 *   `prob < negThreshold` lasting `minSilenceDurationMs` ends it;
 * - `minSpeechDurationMs`: a segment is reported once it is certain to be long
 *   enough, i.e. exactly the segments the offline pass keeps;
 * - `speechPadMs`: starts and ends are padded; where the paddings of two
 *   segments would meet, the pause between them is split in half. An end is
 *   therefore reported once that is decided: when the next segment is
 *   confirmed, or `2 * speechPadMs` into the pause;
 * - `maxSpeechDurationS`: a longer segment is split, at the start of the
 *   current pause if there is one, else right away (the next segment then
 *   starts at the split). Unlike the offline pass it cannot go back to the
 *   longest earlier pause, since that audio is out.
 *
 * Apart from the split rule, the segments equal the offline result.
 */
export class StreamingVadSegmenter {
  private readonly hopSamples: number

  private readonly sampleRate: number

  private threshold = 0.5

  private negThreshold = 0.35

  private minSpeechSamples = 0

  private minSilenceSamples = 0

  private padSamples = 0

  private maxSpeechSamples = Infinity

  /** Index of the next frame */
  private frame = 0

  private triggered = false

  private confirmed = false

  /** Raw start of the current segment (frame start, in samples) */
  private start = 0

  private startProb = 0

  /** Start of the current pause inside the segment, -1 if none */
  private tempEnd = -1

  /** Padded end of the previous segment */
  private prevEnd = 0

  /**
   * The previous segment ended at `pauseStart`, but its padded end is open:
   * a segment starting within 2 * pad of the pause takes half of the pause.
   */
  private pendingEnd: { pauseStart: number; prob: number } | null = null

  /** Total audio length once known (flush), bounds the last partial hop */
  private audioEnd = Infinity

  constructor(options: VadOptions, hopSamples: number, sampleRate = 16000) {
    this.hopSamples = hopSamples
    this.sampleRate = sampleRate
    this.setOptions(options)
  }

  setOptions(options: VadOptions): void {
    const ms = (value: number) => (value * this.sampleRate) / 1000
    this.threshold = options.threshold ?? 0.5
    this.negThreshold =
      options.negThreshold != null && options.negThreshold >= 0
        ? options.negThreshold
        : Math.max(this.threshold - 0.15, 0.01)
    this.minSpeechSamples = ms(options.minSpeechDurationMs ?? 250)
    this.minSilenceSamples = ms(options.minSilenceDurationMs ?? 100)
    this.padSamples = Math.floor(ms(options.speechPadMs ?? 30))
    const maxS = options.maxSpeechDurationS ?? 0
    // Same budget as the offline pass: the padded segment stays within the limit
    this.maxSpeechSamples =
      maxS > 0
        ? maxS * this.sampleRate - this.hopSamples - 2 * this.padSamples
        : Infinity
  }

  /** Whether a confirmed speech segment is open (its end not reported yet) */
  get inSpeech(): boolean {
    return (this.triggered && this.confirmed) || this.pendingEnd !== null
  }

  /** Samples covered by the probabilities consumed so far */
  get processedSamples(): number {
    return this.frame * this.hopSamples
  }

  /**
   * End of the audio that surely belongs to the open speech segment (padded).
   * Only meaningful while `inSpeech`.
   */
  get speechCertainUntil(): number {
    if (this.pendingEnd) {
      const { pauseStart } = this.pendingEnd
      const nextStart = this.triggered ? this.start : this.processedSamples
      return this.earliestPaddedEnd(pauseStart, nextStart)
    }
    // The pause that ends the segment starts here or later, and the next
    // segment starts at least minSilence after it: a pause shorter than
    // 2 * pad gives the segment only half of it, not the full padding.
    const pauseStart = this.tempEnd >= 0 ? this.tempEnd : this.processedSamples
    const nextStart = Math.max(this.processedSamples, pauseStart + this.minSilenceSamples)
    // A max-duration split may also end the segment before its padding
    return Math.min(this.earliestPaddedEnd(pauseStart, nextStart), this.nextSplitSample)
  }

  /** Padded end of a segment whose pause starts at `pauseStart`, with the next segment starting no earlier than `nextStart` */
  private earliestPaddedEnd(pauseStart: number, nextStart: number): number {
    return pauseStart + Math.min(this.padSamples, Math.floor((nextStart - pauseStart) / 2))
  }

  /** First frame at which the open segment is split for exceeding maxSpeechDurationS */
  private get nextSplitSample(): number {
    if (!Number.isFinite(this.maxSpeechSamples)) return Infinity
    const hop = this.hopSamples
    return (Math.floor((this.start + this.maxSpeechSamples) / hop) + 1) * hop
  }

  /** Earliest sample a future or pending segment may still need */
  get retainFrom(): number {
    const from = this.triggered ? this.start : this.processedSamples
    return Math.max(0, from - this.padSamples)
  }

  reset(): void {
    this.frame = 0
    this.triggered = false
    this.confirmed = false
    this.start = 0
    this.startProb = 0
    this.tempEnd = -1
    this.prevEnd = 0
    this.pendingEnd = null
    this.audioEnd = Infinity
  }

  /** Consume the next probabilities, one per hop. */
  push(probs: ArrayLike<number>): VadSegmenterEvent[] {
    const events: VadSegmenterEvent[] = []
    for (let k = 0; k < probs.length; k += 1) {
      this.step(probs[k]!, events)
    }
    this.settlePendingEnd(this.processedSamples, events)
    return events
  }

  /**
   * End of audio after `totalSamples` samples: consumes the last
   * probabilities (the final hop may be partial) and closes open speech.
   */
  flush(totalSamples: number, probs: ArrayLike<number> = []): VadSegmenterEvent[] {
    this.audioEnd = totalSamples
    const events = this.push(probs)
    if (this.triggered) {
      if (this.confirmed || totalSamples - this.start > this.minSpeechSamples) {
        this.confirm(events)
        events.push({ type: 'end', sample: totalSamples, prob: 0 })
        this.close(totalSamples)
      } else {
        this.triggered = false
        this.tempEnd = -1
      }
    }
    if (this.pendingEnd) {
      // No next segment: full padding, stopping at the end of the audio
      this.emitPendingEnd(this.pendingEnd.pauseStart + this.padSamples, events)
    }
    return events
  }

  private step(prob: number, events: VadSegmenterEvent[]): void {
    const s = this.frame * this.hopSamples
    this.frame += 1
    this.settlePendingEnd(s, events)

    if (prob >= this.threshold && this.tempEnd >= 0) {
      this.tempEnd = -1
    }
    if (prob >= this.threshold && !this.triggered) {
      this.triggered = true
      this.confirmed = false
      this.start = s
      this.startProb = prob
      this.maybeConfirm(s + this.hopSamples, events)
      return
    }
    if (this.triggered && s - this.start > this.maxSpeechSamples) {
      // Split. Mid-pause the segment ends where the pause began (padded, as
      // the pause continues); otherwise the next one picks up right here.
      const end =
        this.tempEnd >= 0
          ? Math.min(this.tempEnd + this.padSamples, s)
          : s
      this.confirm(events)
      events.push({ type: 'end', sample: end, prob })
      this.close(end)
      if (prob >= this.threshold) {
        // Still speech: the next segment continues seamlessly from here
        this.triggered = true
        this.start = s
        this.startProb = prob
      }
      return
    }
    if (prob < this.negThreshold && this.triggered) {
      if (this.tempEnd < 0) this.tempEnd = s
      if (s - this.tempEnd < this.minSilenceSamples) {
        this.maybeConfirm(this.tempEnd, events)
        return
      }
      if (this.tempEnd - this.start > this.minSpeechSamples) {
        this.confirm(events)
        this.pendingEnd = { pauseStart: this.tempEnd, prob }
        this.triggered = false
        this.confirmed = false
        this.tempEnd = -1
        this.settlePendingEnd(this.processedSamples, events)
      } else {
        // Too short: never reported. Nothing was emitted, so prevEnd stays.
        this.triggered = false
        this.tempEnd = -1
      }
      return
    }
    if (this.triggered) {
      this.maybeConfirm(this.tempEnd >= 0 ? this.tempEnd : s + this.hopSamples, events)
    }
  }

  // The segment will be kept once its end can no longer fall within
  // minSpeech of its start; `knownEnd` is a lower bound of that end.
  private maybeConfirm(knownEnd: number, events: VadSegmenterEvent[]): void {
    const end = Math.min(knownEnd, this.audioEnd)
    if (!this.confirmed && end - this.start > this.minSpeechSamples) {
      this.confirm(events)
    }
  }

  private confirm(events: VadSegmenterEvent[]): void {
    if (this.confirmed) return
    this.confirmed = true
    let pad = this.padSamples
    if (this.pendingEnd) {
      // Within 2 * pad of the previous segment (else it would be settled):
      // the two share the pause
      pad = Math.floor((this.start - this.pendingEnd.pauseStart) / 2)
      this.emitPendingEnd(this.pendingEnd.pauseStart + pad, events)
    }
    events.push({
      type: 'start',
      sample: Math.max(this.start - pad, this.prevEnd, 0),
      prob: this.startProb,
    })
  }

  /** The pending end gets the full padding once no segment can start within 2 * pad of the pause */
  private settlePendingEnd(nextStart: number, events: VadSegmenterEvent[]): void {
    if (
      this.pendingEnd &&
      !this.triggered &&
      nextStart - this.pendingEnd.pauseStart >= 2 * this.padSamples
    ) {
      this.emitPendingEnd(this.pendingEnd.pauseStart + this.padSamples, events)
    }
  }

  private emitPendingEnd(sample: number, events: VadSegmenterEvent[]): void {
    if (!this.pendingEnd) return
    const end = Math.min(sample, this.audioEnd)
    events.push({ type: 'end', sample: end, prob: this.pendingEnd.prob })
    this.prevEnd = end
    this.pendingEnd = null
  }

  private close(end: number): void {
    this.triggered = false
    this.confirmed = false
    this.tempEnd = -1
    this.prevEnd = end
  }
}
