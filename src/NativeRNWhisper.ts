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
}

export type TranscribeResult = {
  result: string
  language: string
  segments: Array<{
    text: string
    t0: number
    t1: number
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
}

export type NativeWhisperContext = {
  contextPtr: number
  contextId: number
  gpu: boolean
  reasonNoGPU: string
}

export type VadOptions = {
  /** Probability threshold to consider as speech (Default: 0.5) */
  threshold?: number
  /** Min duration for a valid speech segment in ms (Default: 250) */
  minSpeechDurationMs?: number
  /** Min silence duration to consider speech as ended in ms (Default: 100) */
  minSilenceDurationMs?: number
  /** Max duration of a speech segment before forcing a new segment in seconds (Default: 30) */
  maxSpeechDurationS?: number
  /** Padding added before and after speech segments in ms (Default: 30) */
  speechPadMs?: number
  /** Overlap in seconds when copying audio samples from speech segment (Default: 0.1) */
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
}

export type VadSegment = {
  t0: number
  t1: number
}

export interface Spec extends TurboModule {
  install(): Promise<boolean>
}

export default TurboModuleRegistry.get<Spec>('RNWhisper') as Spec
