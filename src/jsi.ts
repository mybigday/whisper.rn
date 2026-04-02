/* eslint-disable no-var */
import type {
  NativeContextOptions,
  NativeWhisperContext,
  NativeVadContextOptions,
  NativeWhisperVadContext,
  TranscribeOptions,
  TranscribeResult,
  VadOptions,
  VadSegment,
} from './NativeRNWhisper'

type TranscribeCallbacks = {
  jobId?: number
  onProgress?: (progress: number) => void
  onNewSegments?: (result: {
    nNew: number
    totalNNew: number
    result: string
    segments: TranscribeResult['segments']
  }) => void
}

declare global {
  var whisperGetConstants: () => Promise<{
    useCoreML: boolean
    coreMLAllowFallback: boolean
  }>
  var whisperInitContext: (
    contextId: number,
    options: NativeContextOptions,
  ) => Promise<NativeWhisperContext>
  var whisperReleaseContext: (contextId: number) => Promise<void>
  var whisperReleaseAllContexts: () => Promise<void>
  var whisperTranscribeFile: (
    contextId: number,
    pathOrBase64: string,
    options: TranscribeOptions & TranscribeCallbacks,
  ) => Promise<TranscribeResult>
  var whisperTranscribeData: (
    contextId: number,
    options: TranscribeOptions & TranscribeCallbacks,
    data: ArrayBuffer,
  ) => Promise<TranscribeResult>
  var whisperAbortTranscribe: (
    contextId: number,
    jobId: number,
  ) => Promise<void>
  var whisperBench: (contextId: number, maxThreads: number) => Promise<string>
  var whisperInitVadContext: (
    contextId: number,
    options: NativeVadContextOptions,
  ) => Promise<NativeWhisperVadContext>
  var whisperReleaseVadContext: (contextId: number) => Promise<void>
  var whisperReleaseAllVadContexts: () => Promise<void>
  var whisperVadDetectSpeech: (
    contextId: number,
    options: VadOptions,
    audioData: ArrayBuffer,
  ) => Promise<{ hasSpeech: boolean; segments: VadSegment[] }>
  var whisperVadDetectSpeechFile: (
    contextId: number,
    pathOrBase64: string,
    options: VadOptions,
  ) => Promise<{ hasSpeech: boolean; segments: VadSegment[] }>
  var whisperToggleNativeLog: (
    enabled: boolean,
    onLog?: (level: string, text: string) => void,
  ) => Promise<void>
}

export {}
