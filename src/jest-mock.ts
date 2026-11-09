import { NativeModules } from 'react-native'

const transcribeResult = {
  language: 'en',
  result: ' Test',
  segments: [{ text: ' Test', t0: 0, t1: 33 }],
  isAborted: false,
}

const parakeetTranscribeResult = {
  language: '',
  result: ' Parakeet test',
  segments: [{ text: ' Parakeet test', t0: 0, t1: 1101 }],
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
global.parakeetInitContext = jest.fn(async (contextId: number) => ({
  contextId,
  gpu: false,
  reasonNoGPU: 'Mock Parakeet context',
}))
global.parakeetReleaseContext = jest.fn(async () => undefined)
global.parakeetReleaseAllContexts = jest.fn(async () => undefined)
global.parakeetTranscribeFile = jest.fn(async () => parakeetTranscribeResult)
global.parakeetTranscribeData = jest.fn(async () => parakeetTranscribeResult)
global.parakeetAbortTranscribe = jest.fn(async () => undefined)
global.whisperInitVadContext = jest.fn(async (contextId: number) => ({
  contextId,
  gpu: false,
  reasonNoGPU: 'Mock VAD context',
  sampleRate: 16000,
  hopSamples: 512,
  arch: 'silero',
}))
global.whisperReleaseVadContext = jest.fn(async () => undefined)
global.whisperReleaseAllVadContexts = jest.fn(async () => undefined)
global.whisperVadDetectSpeech = jest.fn(async () => vadResult)
global.whisperVadDetectSpeechFile = jest.fn(async () => vadResult)

// Mock VAD streams: one probability per 512-sample hop of PCM16 audio, 0.9 when
// the hop's mean absolute amplitude exceeds 1000, else 0.05.
const mockVadHop = 512
const mockVadStreams = new Map<number, number[]>()
let mockVadNextStreamId = 1
const mockVadProbs = (pending: number[], final: boolean) => {
  const probs: number[] = []
  while (pending.length >= mockVadHop || (final && pending.length > 0)) {
    const hop = pending.splice(0, mockVadHop)
    const mean = hop.reduce((sum, v) => sum + Math.abs(v), 0) / mockVadHop
    probs.push(mean > 1000 ? 0.9 : 0.05)
  }
  return probs
}
global.whisperVadStreamCreate = jest.fn(async () => {
  const id = mockVadNextStreamId
  mockVadNextStreamId += 1
  mockVadStreams.set(id, [])
  return id
})
global.whisperVadStreamPush = jest.fn(
  async (_contextId: number, streamId: number, audio: ArrayBuffer) => {
    const pending = mockVadStreams.get(streamId)
    if (!pending) throw new Error('VAD stream not found')
    pending.push(...Array.from(new Int16Array(audio, 0, Math.floor(audio.byteLength / 2))))
    return mockVadProbs(pending, false)
  },
)
global.whisperVadStreamFlush = jest.fn(
  async (_contextId: number, streamId: number) =>
    mockVadProbs(mockVadStreams.get(streamId) ?? [], true),
)
global.whisperVadStreamReset = jest.fn(
  async (_contextId: number, streamId: number) => {
    mockVadStreams.set(streamId, [])
  },
)
global.whisperVadStreamRelease = jest.fn(
  async (_contextId: number, streamId: number) => {
    mockVadStreams.delete(streamId)
  },
)
global.whisperToggleNativeLog = jest.fn(async () => undefined)

module.exports = jest.requireActual('./index')
