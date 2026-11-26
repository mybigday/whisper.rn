import type { VadOptions } from '../index'

/** Why a speech segment ended */
export type VadSegmentEndReason =
  /** A pause of `minSilenceDurationMs` */
  | 'pause'
  /** Split for exceeding `maxSpeechDurationS` */
  | 'max_duration'
  /** The audio ended (flush) */
  | 'end_of_stream'

export type VadSegmenterEvent =
  | {
      /**
       * A speech segment is confirmed. `sample` is its padded start. After a
       * max-duration split it can lie before audio already consumed.
       */
      type: 'start'
      sample: number
      /** Probability of the frame that triggered the segment */
      prob: number
    }
  | {
      /**
       * The current speech segment ended. `sample` is its padded end
       * (exclusive). After a max-duration split it can lie before audio
       * already consumed: the split went back to an earlier pause.
       */
      type: 'end'
      sample: number
      /** Probability of the frame that decided the end */
      prob: number
      reason: VadSegmentEndReason
      /**
       * The split fell inside speech (no pause to cut at): the next segment
       * starts right here, and a word may straddle the cut.
       */
      inSpeech?: boolean
    }

/**
 * Turns per-hop speech probabilities into segment events. `StreamVad` (and so
 * `RealtimeTranscriber`) runs one; `StreamingVadSegmenter` is the default.
 */
export interface SpeechSegmenter {
  push(probs: ArrayLike<number>): VadSegmenterEvent[]
  /** End of audio after `totalSamples` samples, with the last probabilities */
  flush(totalSamples: number, probs?: ArrayLike<number>): VadSegmenterEvent[]
  reset(): void
  setOptions(options: VadOptions): void
  /** Audio up to here surely belongs to the open segment (emitted as it comes) */
  readonly speechCertainUntil: number
  /** Earliest sample a future or pending segment may still need */
  readonly retainFrom: number
  /** Whether an end or next start may still move back into consumed audio */
  readonly maySplitBack: boolean
  /** Optional: energy of audio samples, for picking split points */
  setFrameEnergy?(frameEnergy: ((start: number, end: number) => number) | undefined): void
}

export type StreamingVadSegmenterOptions = {
  /**
   * Where a segment without any usable pause is split at `maxSpeechDurationS`.
   * - true (default): at the least speech-like frame of the last 3 s (0.5 s
   *   before the limit at the latest, and in the second half of the segment),
   *   the quietest one per `frameEnergy` when several are about as low. An ongoing pause longer than every earlier one
   *   ends the segment where it began.
   * - false: right at the limit, exactly as the offline pass does.
   */
  splitAtDip?: boolean
  /**
   * Energy (e.g. mean square) of the audio samples `[start, end)`, used to pick
   * the quietest of the near-lowest frames for a dip split. Without it, the
   * latest frame with the lowest probability wins.
   */
  frameEnergy?: (start: number, end: number) => number
}

/** silero-vad: pauses longer than this are split candidates at max duration */
const MIN_SILENCE_AT_MAX_SPEECH_MS = 98

/** Dip split: frames considered, and the end of the window before the limit */
const DIP_WINDOW_MS = 3000
const DIP_GUARD_MS = 500
/** Dip split: frames this close to the lowest probability compete on energy */
const DIP_PROB_TIE = 0.05

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
 * - `maxSpeechDurationS`: a longer segment is split, like offline, at the
 *   longest earlier pause of more than 98 ms; the next segment starts where
 *   that pause ended. Both events then lie in the past, so the consumer must
 *   keep the open segment's audio (from `retainFrom`) to cut it there.
 *   Without such a pause see `StreamingVadSegmenterOptions.splitAtDip`.
 *
 * With `splitAtDip: false` the segments equal the offline result.
 */
export class StreamingVadSegmenter implements SpeechSegmenter {
  private readonly hopSamples: number

  private readonly sampleRate: number

  private readonly splitAtDip: boolean

  private frameEnergy?: (start: number, end: number) => number

  private threshold = 0.5

  private negThreshold = 0.35

  private minSpeechSamples = 0

  private minSilenceSamples = 0

  private padSamples = 0

  private maxSpeechSamples = Infinity

  private readonly minSilenceAtMaxSamples: number

  /** Index of the next frame */
  private frame = 0

  private triggered = false

  private confirmed = false

  /** Raw start of the current segment (in samples; frame aligned unless it began at a dip split) */
  private start = 0

  private startProb = 0

  /** Start of the current pause inside the segment, -1 if none */
  private tempEnd = -1

  /** Pauses of the current segment that may take a max-duration split */
  private possibleEnds: Array<{ start: number; length: number }> = []

  /** Probabilities of the current segment's frames, from frame `historyBase` */
  private history: number[] = []

  private historyBase = 0

  /** Padded end of the previous segment */
  private prevEnd = 0

  /**
   * The previous segment ended at `pauseStart`, but its padded end is open:
   * a segment starting within 2 * pad of the pause takes half of the pause.
   */
  private pendingEnd: {
    pauseStart: number
    prob: number
    reason: VadSegmentEndReason
    inSpeech?: boolean
  } | null = null

  /** Total audio length once known (flush), bounds the last partial hop */
  private audioEnd = Infinity

  constructor(
    options: VadOptions,
    hopSamples: number,
    sampleRate = 16000,
    segmenterOptions: StreamingVadSegmenterOptions = {},
  ) {
    this.hopSamples = hopSamples
    this.sampleRate = sampleRate
    this.splitAtDip = segmenterOptions.splitAtDip ?? true
    this.frameEnergy = segmenterOptions.frameEnergy
    this.minSilenceAtMaxSamples = (MIN_SILENCE_AT_MAX_SPEECH_MS * sampleRate) / 1000
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
   * End of the audio that surely belongs to the open speech segment (padded),
   * unless a max-duration split later moves the end back.
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

  /**
   * Earliest sample a future or pending segment may still need. While a
   * segment that may be split is open, that is all of it: a split can move
   * both its end and the next start back into audio already consumed.
   */
  get retainFrom(): number {
    const from = this.triggered ? this.start : this.processedSamples
    return Math.max(0, from - this.padSamples)
  }

  /**
   * Whether segment bounds may still move back into audio already consumed:
   * an open segment may be split, or the end of one split at a pause in
   * progress waits for the next start, which can come sooner than
   * `minSilenceDurationMs` and take part of the padding already consumed.
   */
  get maySplitBack(): boolean {
    return (
      (this.triggered && Number.isFinite(this.maxSpeechSamples)) ||
      this.pendingEnd?.reason === 'max_duration'
    )
  }

  /** Set or replace the energy function of `StreamingVadSegmenterOptions.frameEnergy` */
  setFrameEnergy(frameEnergy: ((start: number, end: number) => number) | undefined): void {
    this.frameEnergy = frameEnergy
  }

  reset(): void {
    this.frame = 0
    this.triggered = false
    this.confirmed = false
    this.start = 0
    this.startProb = 0
    this.tempEnd = -1
    this.possibleEnds = []
    this.history = []
    this.historyBase = 0
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
        events.push({ type: 'end', sample: totalSamples, prob: 0, reason: 'end_of_stream' })
        this.close(totalSamples)
      } else {
        this.untrigger()
      }
    }
    if (this.pendingEnd) {
      // No next segment: full padding, stopping at the end of the audio
      this.emitPendingEnd(this.pendingEnd.pauseStart + this.padSamples, events)
    }
    return events
  }

  private step(prob: number, events: VadSegmenterEvent[]): void {
    const { frame } = this
    const s = frame * this.hopSamples
    this.frame += 1
    this.settlePendingEnd(s, events)
    if (this.triggered) this.history.push(prob)

    if (prob >= this.threshold && this.tempEnd >= 0) {
      const length = s - this.tempEnd
      if (length > this.minSilenceAtMaxSamples) {
        this.possibleEnds.push({ start: this.tempEnd, length })
      }
      this.tempEnd = -1
    }
    if (prob >= this.threshold && !this.triggered) {
      this.triggered = true
      this.confirmed = false
      this.start = s
      this.startProb = prob
      this.possibleEnds = []
      this.history = [prob]
      this.historyBase = frame
      this.maybeConfirm(s + this.hopSamples, events)
      return
    }
    if (this.triggered && s - this.start > this.maxSpeechSamples) {
      if (this.split(s, prob, events)) return
    }
    if (prob < this.negThreshold && this.triggered) {
      if (this.tempEnd < 0) this.tempEnd = s
      if (s - this.tempEnd < this.minSilenceSamples) {
        this.maybeConfirm(this.tempEnd, events)
        return
      }
      if (this.tempEnd - this.start > this.minSpeechSamples) {
        this.confirm(events)
        this.pendingEnd = { pauseStart: this.tempEnd, prob, reason: 'pause' }
        this.untrigger()
        this.settlePendingEnd(this.processedSamples, events)
      } else {
        // Too short: never reported. Nothing was emitted, so prevEnd stays.
        this.untrigger()
      }
      return
    }
    if (this.triggered) {
      this.maybeConfirm(this.tempEnd >= 0 ? this.tempEnd : s + this.hopSamples, events)
    }
  }

  /**
   * Split the open segment for exceeding maxSpeechDurationS at frame `s`.
   * Returns true when the frame is fully handled (no segment continues).
   */
  private split(s: number, prob: number, events: VadSegmenterEvent[]): boolean {
    this.confirm(events)

    // The longest earlier pause; the first one wins a tie, as offline
    const best = this.possibleEnds.reduce<{ start: number; length: number } | null>(
      (acc, pause) => (!acc || pause.length > acc.length ? pause : acc),
      null,
    )
    const ongoing = this.tempEnd >= 0 ? s - this.tempEnd : 0
    if (this.splitAtDip && ongoing > 0 && (!best || ongoing > best.length)) {
      // The pause in progress is the best cut: end where it began, the next
      // segment starts once speech resumes
      this.pendingEnd = { pauseStart: this.tempEnd, prob, reason: 'max_duration' }
      this.untrigger()
      return true
    }

    if (best) {
      // As offline: end at the pause, continue from where speech resumed. The
      // padding between the two is settled like any other pause.
      this.pendingEnd = { pauseStart: best.start, prob, reason: 'max_duration' }
      this.restartAt(best.start + best.length)
      this.settlePendingEnd(s, events)
      return false
    }

    // No dip window (a short limit) but speech goes on: cut here, seamlessly. The
    // offline rule below would start the next segment a frame later, dropping
    // that frame when there is no padding to share.
    const dip = this.splitAtDip ? this.findDip(s) : -1
    const cut = dip < 0 && this.splitAtDip && prob >= this.threshold ? s : dip
    if (cut >= 0) {
      // Mid-speech: no pause to pad into, the two segments meet at the cut
      events.push({ type: 'end', sample: cut, prob, reason: 'max_duration', inSpeech: true })
      this.prevEnd = cut
      this.restartAt(cut)
      return false
    }

    // As offline: end right here; the next segment starts at the next speech frame
    this.pendingEnd = {
      pauseStart: s,
      prob,
      reason: 'max_duration',
      inSpeech: prob >= this.negThreshold,
    }
    this.untrigger()
    return true
  }

  /** Continue speech as a new, unconfirmed segment from sample `start` */
  private restartAt(start: number): void {
    const frame = Math.floor(start / this.hopSamples)
    const drop = Math.max(0, Math.min(frame - this.historyBase, this.history.length))
    this.history = this.history.slice(drop)
    this.historyBase += drop
    this.triggered = true
    this.confirmed = false
    this.start = start
    this.startProb = this.history[0] ?? this.startProb
    this.tempEnd = -1
    this.possibleEnds = []
  }

  /**
   * Dip split point for a segment hitting the limit at frame `s`: the middle
   * of the least speech-like frame in the window, -1 if the window is empty.
   */
  private findDip(s: number): number {
    const hop = this.hopSamples
    const ms = (value: number) => (value * this.sampleRate) / 1000
    // In the second half of the segment only: a cut near its start would leave
    // a remainder that is over the limit again right away (and a tiny segment)
    const from = Math.max(
      this.historyBase,
      Math.ceil(Math.max(this.start + this.maxSpeechSamples / 2, s - ms(DIP_WINDOW_MS)) / hop),
    )
    // Frames that end by the guard, excluding the current one (s)
    const to = Math.min(
      Math.floor((s - ms(DIP_GUARD_MS)) / hop),
      this.historyBase + this.history.length - 1,
    )
    if (to <= from) return -1

    let lowest = Infinity
    for (let f = from; f < to; f += 1) {
      lowest = Math.min(lowest, this.history[f - this.historyBase]!)
    }
    let bestFrame = -1
    let bestEnergy = Infinity
    for (let f = from; f < to; f += 1) {
      const p = this.history[f - this.historyBase]!
      if (this.frameEnergy) {
        if (p <= lowest + DIP_PROB_TIE) {
          const energy = this.frameEnergy(f * hop, (f + 1) * hop)
          // Ties go to the later frame: less to carry into the next segment
          if (energy <= bestEnergy) {
            bestEnergy = energy
            bestFrame = f
          }
        }
      } else if (p <= lowest) {
        bestFrame = f
      }
    }
    return bestFrame * hop + Math.floor(hop / 2)
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
      // The two share the pause where their paddings would meet
      pad = Math.min(pad, Math.floor((this.start - this.pendingEnd.pauseStart) / 2))
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
    if (!this.pendingEnd) return
    // An open (not yet confirmed) segment already fixes where the next one starts
    const next = this.triggered ? this.start : nextStart
    if (next - this.pendingEnd.pauseStart >= 2 * this.padSamples) {
      this.emitPendingEnd(this.pendingEnd.pauseStart + this.padSamples, events)
    }
  }

  private emitPendingEnd(sample: number, events: VadSegmenterEvent[]): void {
    if (!this.pendingEnd) return
    const end = Math.min(sample, this.audioEnd)
    const { prob, reason, inSpeech } = this.pendingEnd
    events.push({ type: 'end', sample: end, prob, reason, ...(inSpeech ? { inSpeech } : {}) })
    this.prevEnd = end
    this.pendingEnd = null
  }

  private untrigger(): void {
    this.triggered = false
    this.confirmed = false
    this.tempEnd = -1
    this.possibleEnds = []
    this.history = []
  }

  private close(end: number): void {
    this.untrigger()
    this.prevEnd = end
  }
}
