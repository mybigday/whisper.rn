export type LanguageLockOptions = {
  /** Lock once a language was detected this many times with `minProb` (Default: 3) */
  lockAfter?: number
  /** Detections below this probability do not count (Default: 0.5) */
  minProb?: number
  /** While locked, auto-detect again on every Nth final (Default: 8) */
  recheckEvery?: number
  /** Only re-check on finals at least this long, in ms (Default: 3000) */
  recheckMinMs?: number
  /** Unlock after this many re-checks in a row confidently detect another language (Default: 2) */
  unlockAfter?: number
  /** Probability a re-check needs to count against the lock (Default: 0.9) */
  unlockProb?: number
}

/**
 * Keeps an auto-detected source language stable across slices: per-slice
 * detection on short audio flips between languages, and every flip changes
 * the transcript. Once a language was detected confidently `lockAfter` times
 * it is passed explicitly (which also skips detection); occasional re-checks
 * on long finals let a real language change unlock it.
 */
export class LanguageLock {
  private readonly options: Required<LanguageLockOptions>

  private counts = new Map<string, number>()

  private lockedLanguage: string | undefined

  private finalsSinceCheck = 0

  private disagreements = 0

  constructor(options: LanguageLockOptions = {}) {
    this.options = {
      lockAfter: options.lockAfter ?? 3,
      minProb: options.minProb ?? 0.5,
      recheckEvery: options.recheckEvery ?? 8,
      recheckMinMs: options.recheckMinMs ?? 3000,
      unlockAfter: options.unlockAfter ?? 2,
      unlockProb: options.unlockProb ?? 0.9,
    }
  }

  /** The locked language, if any */
  get locked(): string | undefined {
    return this.lockedLanguage
  }

  /**
   * Language to request for a transcription: the locked one, or undefined to
   * auto-detect (no lock yet, or a final due for a re-check).
   */
  languageFor(isFinal: boolean, durationMs: number): string | undefined {
    if (!this.lockedLanguage) return undefined
    if (!isFinal) return this.lockedLanguage
    this.finalsSinceCheck += 1
    if (
      this.finalsSinceCheck >= this.options.recheckEvery &&
      durationMs >= this.options.recheckMinMs
    ) {
      this.finalsSinceCheck = 0
      return undefined
    }
    return this.lockedLanguage
  }

  /** A final's auto-detected language (only call it when the language was detected) */
  observe(language: string, prob: number | undefined): void {
    if (!language) return
    const p = prob ?? 0
    if (!this.lockedLanguage) {
      if (p < this.options.minProb) return
      const count = (this.counts.get(language) ?? 0) + 1
      this.counts.set(language, count)
      if (count >= this.options.lockAfter) {
        this.lockedLanguage = language
        this.finalsSinceCheck = 0
        this.disagreements = 0
      }
      return
    }
    if (language !== this.lockedLanguage && p >= this.options.unlockProb) {
      this.disagreements += 1
      if (this.disagreements >= this.options.unlockAfter) {
        this.reset()
        // The confident new language starts the next count
        this.counts.set(language, 1)
      }
    } else {
      this.disagreements = 0
    }
  }

  reset(): void {
    this.counts.clear()
    this.lockedLanguage = undefined
    this.finalsSinceCheck = 0
    this.disagreements = 0
  }
}
