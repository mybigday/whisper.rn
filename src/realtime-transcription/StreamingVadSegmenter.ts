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
 * - `speechPadMs`: starts and ends are padded, a start never reaches back
 *   before the previous segment's end;
 * - `maxSpeechDurationS`: a longer segment is split, at the start of the
 *   current pause if there is one, else right away (the next segment then
 *   starts at the split). Unlike the offline pass it cannot go back to the
 *   longest earlier pause, since that audio is out.
 *
 * Apart from the split rule, and where two segments' paddings meet (the
 * offline pass splits the gap between them in half, this gives it to the
 * earlier one), the segments equal the offline result.
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

  /** End decided but not reported yet: its padding reaches past the audio seen so far */
  private pendingEnd: { sample: number; prob: number } | null = null

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
    if (this.pendingEnd) return this.pendingEnd.sample
    const knownEnd = this.tempEnd >= 0 ? this.tempEnd : this.processedSamples
    return knownEnd + this.padSamples
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
  }

  /** Consume the next probabilities, one per hop. */
  push(probs: ArrayLike<number>): VadSegmenterEvent[] {
    const events: VadSegmenterEvent[] = []
    for (let k = 0; k < probs.length; k += 1) {
      this.step(probs[k]!, events)
    }
    if (this.pendingEnd && this.pendingEnd.sample <= this.processedSamples) {
      this.emitPendingEnd(events)
    }
    return events
  }

  /** End of audio after `totalSamples` samples: closes an open segment. */
  flush(totalSamples: number): VadSegmenterEvent[] {
    const events: VadSegmenterEvent[] = []
    if (this.pendingEnd) {
      // Like offline, the padding stops at the end of the audio
      this.pendingEnd.sample = Math.min(this.pendingEnd.sample, totalSamples)
      this.emitPendingEnd(events)
    }
    if (this.triggered) {
      if (totalSamples - this.start > this.minSpeechSamples) {
        this.confirm(events)
        events.push({ type: 'end', sample: totalSamples, prob: 0 })
      }
      this.close(totalSamples)
    }
    return events
  }

  private step(prob: number, events: VadSegmenterEvent[]): void {
    const s = this.frame * this.hopSamples
    this.frame += 1

    if (prob >= this.threshold && this.tempEnd >= 0) {
      this.tempEnd = -1
    }
    if (prob >= this.threshold && !this.triggered) {
      // The previous segment's end is final once the next one begins
      this.emitPendingEnd(events)
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
        const end = this.tempEnd + this.padSamples
        this.pendingEnd = { sample: end, prob }
        if (end <= this.processedSamples) this.emitPendingEnd(events)
        this.close(end)
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
    if (!this.confirmed && knownEnd - this.start > this.minSpeechSamples) {
      this.confirm(events)
    }
  }

  private confirm(events: VadSegmenterEvent[]): void {
    if (this.confirmed) return
    this.confirmed = true
    events.push({
      type: 'start',
      sample: Math.max(this.start - this.padSamples, this.prevEnd, 0),
      prob: this.startProb,
    })
  }

  private emitPendingEnd(events: VadSegmenterEvent[]): void {
    if (!this.pendingEnd) return
    events.push({ type: 'end', ...this.pendingEnd })
    this.prevEnd = this.pendingEnd.sample
    this.pendingEnd = null
  }

  private close(end: number): void {
    this.triggered = false
    this.confirmed = false
    this.tempEnd = -1
    this.prevEnd = end
  }
}
