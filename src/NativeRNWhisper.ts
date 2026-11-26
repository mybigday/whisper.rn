import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

// Common transcribe options
export type TranscribeOptions = {
  /** Spoken language (Default: 'auto' for auto-detect) */
  language?: string
  /** Translate from source language to english (Default: false) */
  translate?: boolean
  /** Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores) */
  maxThreads?: number
  /** Number of processors to use for parallel processing with whisper_full_parallel (Default: 1 to use whisper_full) */
  nProcessors?: number
  /** Maximum number of text context tokens to store */
  maxContext?: number
  /** Maximum segment length in characters */
  maxLen?: number
  /** Enable token-level timestamps */
  tokenTimestamps?: boolean
  /** Enable tinydiarize (requires a tdrz model) */
  tdrzEnable?: boolean
  /** Word timestamp probability threshold */
  wordThold?: number
  /** Time offset in milliseconds */
  offset?: number
  /** Duration of audio to process in milliseconds */
  duration?: number
  /** Initial decoding temperature */
  temperature?: number
  /** Temperature fallback increment applied between decoding retries */
  temperatureInc?: number
  /** Beam size for beam search */
  beamSize?: number
  /** Number of best candidates to keep */
  bestOf?: number
  /** Initial Prompt */
  prompt?: string
  /**
   * Add each segment's text tokens (`segments[].tokens`) to the result (Default: false).
   * Token times need `tokenTimestamps` (t0 / t1) or a context initialized with DTW (tDtw).
   */
  includeTokens?: boolean
}

/** A text token of a Whisper result (special and timestamp tokens are left out) */
export type TranscribeToken = {
  id: number
  /**
   * Text of the token. A character split across tokens (common in CJK) comes with
   * the token that completes it, earlier ones have empty text; the texts of a
   * segment's tokens add up to its text.
   */
  text: string
  /** Probability of the token */
  p: number
  /** Log probability of the token */
  plog: number
  /** Start / end in 10 ms units with `tokenTimestamps`, else -1 */
  t0: number
  t1: number
  /** DTW-aligned time in 10 ms units when the context uses DTW (`dtwAheadsPreset`), else -1 */
  tDtw: number
}

export type TranscribeResult = {
  result: string
  language: string
  /** Whisper: probability of the language when it was auto-detected (`language` unset or 'auto') */
  languageProb?: number
  segments: Array<{
    text: string
    t0: number
    t1: number
    /** Whisper: probability that the segment's audio window has no speech */
    noSpeechProb?: number
    /** Whisper: mean log probability of the segment's text tokens (low values suggest hallucination) */
    avgLogprob?: number
    /** Whisper with `includeTokens` */
    tokens?: TranscribeToken[]
  }>
  isAborted: boolean
}

export type CoreMLAsset = {
  uri: string
  filepath: string
}

export type NativeContextOptions = {
  filePath: string
  isBundleAsset: boolean
  useFlashAttn?: boolean
  useGpu?: boolean
  useCoreMLIos?: boolean
  downloadCoreMLAssets?: boolean
  coreMLAssets?: CoreMLAsset[]
  dtwAheadsPreset?: string
  dtwNTop?: number
}

export type NativeWhisperContext = {
  contextPtr: number
  contextId: number
  gpu: boolean
  reasonNoGPU: string
  /** DTW token timestamps are on */
  dtw?: boolean
}

export type NativeParakeetContextOptions = {
  filePath: string
  isBundleAsset: boolean
  useGpu?: boolean
}

export type NativeParakeetContext = {
  contextId: number
  gpu: boolean
  reasonNoGPU: string
}

export type VadOptions = {
  /** Probability threshold to consider as speech (Default: 0.5) */
  threshold?: number
  /**
   * Probability below which audio counts as silence while in speech, the lower
   * edge of the hysteresis (Default: max(threshold - 0.15, 0.01))
   */
  negThreshold?: number
  /** Min duration for a valid speech segment in ms (Default: 250) */
  minSpeechDurationMs?: number
  /** Min silence duration to consider speech as ended in ms (Default: 100) */
  minSilenceDurationMs?: number
  /** Max duration of a speech segment before forcing a new segment in seconds (Default: no limit) */
  maxSpeechDurationS?: number
  /** Padding added before and after speech segments in ms (Default: 30) */
  speechPadMs?: number
  /**
   * @deprecated Ignored. It applied to whisper.cpp's windowed VAD; detection
   * now keeps the model state across the whole input, so nothing overlaps.
   */
  samplesOverlap?: number
}

export type NativeVadContextOptions = {
  filePath: string
  isBundleAsset: boolean
  useGpu?: boolean
  nThreads?: number
}

export type NativeWhisperVadContext = {
  contextId: number
  gpu: boolean
  reasonNoGPU: string
  /** Sample rate the model expects (16000) */
  sampleRate: number
  /** Samples between consecutive speech probabilities (512 for Silero) */
  hopSamples: number
  /** Model architecture, e.g. `silero` */
  arch: string
}

export type VadSegment = {
  t0: number
  t1: number
}

export interface Spec extends TurboModule {
  install(): Promise<boolean>
}

export default TurboModuleRegistry.get<Spec>('RNWhisper') as Spec
