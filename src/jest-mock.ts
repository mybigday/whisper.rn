import { NativeModules } from 'react-native'

const transcribeResult = {
  language: 'en',
  result: ' Test',
  segments: [{ text: ' Test', t0: 0, t1: 33 }],
  isAborted: false,
}

const vadResult = {
  hasSpeech: true,
  segments: [
    { t0: 0.5, t1: 2.3 },
    { t0: 3.1, t1: 5.8 },
    { t0: 7.2, t1: 9.4 },
  ],
}

if (!NativeModules.RNWhisper) {
  NativeModules.RNWhisper = {
    install: jest.fn(async () => true),
    getConstants: jest.fn(() => ({
      useCoreML: false,
      coreMLAllowFallback: false,
    })),
  }
}

global.whisperGetConstants = jest.fn(async () => ({
  useCoreML: false,
  coreMLAllowFallback: false,
}))
global.whisperInitContext = jest.fn(async (contextId: number) => ({
  contextPtr: contextId,
  contextId,
  gpu: false,
  reasonNoGPU: 'Mock context',
}))
global.whisperReleaseContext = jest.fn(async () => undefined)
global.whisperReleaseAllContexts = jest.fn(async () => undefined)
global.whisperTranscribeFile = jest.fn(async () => transcribeResult)
global.whisperTranscribeData = jest.fn(
  async (
    _contextId: number,
    options: { onProgress?: (progress: number) => void },
  ) => {
    options.onProgress?.(100)
    return transcribeResult
  },
)
global.whisperAbortTranscribe = jest.fn(async () => undefined)
global.whisperBench = jest.fn(async () =>
  JSON.stringify(['NEON', 1, 1, 1, 1, 1]),
)
global.whisperInitVadContext = jest.fn(async (contextId: number) => ({
  contextId,
  gpu: false,
  reasonNoGPU: 'Mock VAD context',
}))
global.whisperReleaseVadContext = jest.fn(async () => undefined)
global.whisperReleaseAllVadContexts = jest.fn(async () => undefined)
global.whisperVadDetectSpeech = jest.fn(async () => vadResult)
global.whisperVadDetectSpeechFile = jest.fn(async () => vadResult)
global.whisperToggleNativeLog = jest.fn(async () => undefined)

module.exports = jest.requireActual('./index')
