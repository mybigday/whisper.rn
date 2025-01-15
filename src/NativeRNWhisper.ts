import { TurboModuleRegistry } from 'react-native'
import type { TurboModule } from 'react-native/Libraries/TurboModule/RCTExport'

// Common transcribe options
export type TranscribeOptions = {
  /** Spoken language (Default: 'auto' for auto-detect) */
  language?: string,
  /** Translate from source language to english (Default: false) */
  translate?: boolean,
  /** Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores) */
  maxThreads?: number,
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
  contextId: number
  gpu: boolean
  reasonNoGPU: string
}

export interface Spec extends TurboModule {
  getConstants(): {
    useCoreML: boolean
    coreMLAllowFallback: boolean
  };
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

  // New methods for pausing and resuming transcription
  pauseRealtimeTranscribe(contextId: number): Promise<void>;
  resumeRealtimeTranscribe(contextId: number): Promise<void>;

  abortTranscribe(contextId: number, jobId: number): Promise<void>;

  bench(contextId: number, maxThreads: number): Promise<string>;

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
