import type { TurboModule } from 'react-native/Libraries/TurboModule/RCTExport'
import { TurboModuleRegistry } from 'react-native'

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
  /** Speed up audio by x2 (reduced accuracy) */
  speedUp?: boolean,
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
}

export interface Spec extends TurboModule {
  getConstants(): {
    useCoreML: boolean
    coreMLAllowFallback: boolean
  };
  initContext(filePath: string, isBundleAsset: boolean): Promise<number>;
  releaseContext(contextId: number): Promise<void>;
  releaseAllContexts(): Promise<void>;
  transcribeFile(
    contextId: number,
    jobId: number,
    path: string,
    options: TranscribeOptions,
  ): Promise<TranscribeResult>;
  startRealtimeTranscribe(
    contextId: number,
    jobId: number,
    options: TranscribeOptions,
  ): Promise<void>;
  abortTranscribe(contextId: number, jobId: number): Promise<void>;
}

export default TurboModuleRegistry.get<Spec>('RNWhisper') as Spec
