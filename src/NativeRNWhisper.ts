import type { TurboModule } from 'react-native/Libraries/TurboModule/RCTExport'
import { TurboModuleRegistry } from 'react-native'

// Common transcribe options
export type TranscribeOptions = {
  /** Spoken language (Default: 'auto' for auto-detect) */
  language?: string,
  /** Translate from source language to english (Default: false) */
  translate?: boolean,
  /** Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores) */
  maxThreads?: number,
  /** Number of processors to use for parallel processing with whisper_full_parallel (Default: 1 to use whisper_full) */
  nProcessors?: number,
  /** Maximum number of text context tokens to store */
  maxContext?: number,
  /** Maximum segment length in characters */
  maxLen?: number,
  /** Enable token-level timestamps */
  tokenTimestamps?: boolean,
  /** Enable tinydiarize (requires a tdrz model) */
  tdrzEnable?: boolean,
  /** Word timestamp probability threshold */
  wordThold?: number,
  /** Time offset in milliseconds */
  offset?: number,
  /** Duration of audio to process in milliseconds */
  duration?: number,
  /** Tnitial decoding temperature */
  temperature?: number,
  temperatureInc?: number,
  /** Beam size for beam search */
  beamSize?: number,
  /** Number of best candidates to keep */
  bestOf?: number,
  /** Initial Prompt */
  prompt?: string,
}

export type TranscribeResult = {
  result: string,
  segments: Array<{
    text: string,
    t0: number,
    t1: number,
  }>,
  isAborted: boolean,
}

export type CoreMLAsset = {
  uri: string,
  filepath: string,
}

type NativeContextOptions = {
  filePath: string,
  isBundleAsset: boolean,
  useFlashAttn?: boolean,
  useGpu?: boolean,
  useCoreMLIos?: boolean,
  downloadCoreMLAssets?: boolean,
  coreMLAssets?: CoreMLAsset[],
}

export type NativeWhisperContext = {
  contextPtr: number
  contextId: number
  gpu: boolean
  reasonNoGPU: string
}

export type VadOptions = {
  /** Probability threshold to consider as speech (Default: 0.5) */
  threshold?: number,
  /** Min duration for a valid speech segment in ms (Default: 250) */
  minSpeechDurationMs?: number,
  /** Min silence duration to consider speech as ended in ms (Default: 100) */
  minSilenceDurationMs?: number,
  /** Max duration of a speech segment before forcing a new segment in seconds (Default: 30) */
  maxSpeechDurationS?: number,
  /** Padding added before and after speech segments in ms (Default: 30) */
  speechPadMs?: number,
  /** Overlap in seconds when copying audio samples from speech segment (Default: 0.1) */
  samplesOverlap?: number,
}

type NativeVadContextOptions = {
  filePath: string,
  isBundleAsset: boolean,
  useGpu?: boolean,
  nThreads?: number,
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
  toggleNativeLog(enabled: boolean): Promise<void>

  getConstants(): {
    useCoreML: boolean
    coreMLAllowFallback: boolean
  };
  installJSIBindings(): Promise<{ success: boolean }>;
  initContext(options: NativeContextOptions): Promise<NativeWhisperContext>;
  releaseContext(contextId: number): Promise<void>;
  releaseAllContexts(): Promise<void>;
  transcribeFile(
    contextId: number,
    jobId: number,
    pathOrBase64: string,
    options: {}, // TranscribeOptions & { onProgress?: boolean, onNewSegments?: boolean }
  ): Promise<TranscribeResult>;
  transcribeData(
    contextId: number,
    jobId: number,
    dataBase64: string,
    options: {}, // TranscribeOptions & { onProgress?: boolean, onNewSegments?: boolean }
  ): Promise<TranscribeResult>;
  startRealtimeTranscribe(
    contextId: number,
    jobId: number,
    options: TranscribeOptions,
  ): Promise<void>;
  abortTranscribe(contextId: number, jobId: number): Promise<void>;

  bench(contextId: number, maxThreads: number): Promise<string>;

  // VAD methods
  initVadContext(options: NativeVadContextOptions): Promise<NativeWhisperVadContext>;
  releaseVadContext(contextId: number): Promise<void>;
  releaseAllVadContexts(): Promise<void>;
  vadDetectSpeech(
    contextId: number,
    audioData: string, // base64 encoded float32 PCM data
    options: VadOptions,
  ): Promise<VadSegment[]>;
  vadDetectSpeechFile(
    contextId: number,
    filePathOrBase64: string,
    options: VadOptions,
  ): Promise<VadSegment[]>;

  // iOS specific
  getAudioSessionCurrentCategory: () => Promise<{
    category: string,
    options: Array<string>,
  }>;
  getAudioSessionCurrentMode: () => Promise<string>;
  setAudioSessionCategory: (category: string, options: Array<string>) => Promise<void>;
  setAudioSessionMode: (mode: string) => Promise<void>;
  setAudioSessionActive: (active: boolean) => Promise<void>;
}

export default TurboModuleRegistry.get<Spec>('RNWhisper') as Spec
