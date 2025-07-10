import { NativeModules, DeviceEventEmitter } from 'react-native'

if (!NativeModules.RNWhisper) {
  NativeModules.RNWhisper = {
    installJSIBindings: jest.fn(() => Promise.resolve()),
    initContext: jest.fn(() => Promise.resolve({ contextId: 1 })),
    transcribeFile: jest.fn(() => Promise.resolve({
      result: ' Test',
      segments: [{ text: ' Test', t0: 0, t1: 33 }],
      isAborted: false,
    })),
    transcribeData: jest.fn(() => Promise.resolve({
      result: ' Test',
      segments: [{ text: ' Test', t0: 0, t1: 33 }],
      isAborted: false,
    })),
    startRealtimeTranscribe: jest.fn((contextId, jobId) => {
      setTimeout(() => {
        // Start
        DeviceEventEmitter.emit('@RNWhisper_onRealtimeTranscribe', {
          contextId,
          jobId,
          payload: {
            isCapturing: true,
            data: {
              result: ' Test',
              segments: [{ text: ' Test', t0: 0, t1: 33 }],
            },
            processTime: 100,
            recordingTime: 1000,
          },
        })
        DeviceEventEmitter.emit('@RNWhisper_onRealtimeTranscribe', {
          contextId,
          jobId,
          payload: {
            isCapturing: false,
            data: {
              result: ' Test',
              segments: [{ text: ' Test', t0: 0, t1: 33 }],
            },
            processTime: 100,
            recordingTime: 2000,
          },
        })
        // End event
        DeviceEventEmitter.emit('@RNWhisper_onRealtimeTranscribeEnd', {
          contextId,
          jobId,
          payload: {},
        })
      })
    }),
    bench: jest.fn(() => Promise.resolve({
      config: 'NEON',
      nThreads: 1,
      encodeMs: 1,
      decodeMs: 1,
      batchMs: 1,
      promptMs: 1,
    })),
    releaseContext: jest.fn(() => Promise.resolve()),
    releaseAllContexts: jest.fn(() => Promise.resolve()),

    // VAD methods
    initVadContext: jest.fn(() => Promise.resolve({
      contextId: 2,
      gpu: false,
      reasonNoGPU: 'Mock VAD context'
    })),
    vadDetectSpeech: jest.fn().mockResolvedValue([
      { t0: 0.5, t1: 2.3 },
      { t0: 3.1, t1: 5.8 },
      { t0: 7.2, t1: 9.4 }
    ]),
    vadDetectSpeechFile: jest.fn().mockResolvedValue([
      { t0: 0.5, t1: 2.3 },
      { t0: 3.1, t1: 5.8 },
      { t0: 7.2, t1: 9.4 }
    ]),
    releaseVadContext: jest.fn(() => Promise.resolve()),
    releaseAllVadContexts: jest.fn(() => Promise.resolve()),

    // iOS AudioSession utils
    getAudioSessionCurrentCategory: jest.fn(() => Promise.resolve({
      category: 'AVAudioSessionCategoryPlayAndRecord',
      options: [],
    })),
    getAudioSessionCurrentMode: jest.fn(() => Promise.resolve('')),
    setAudioSessionCategory: jest.fn(() => Promise.resolve()),
    setAudioSessionMode: jest.fn(() => Promise.resolve()),
    setAudioSessionActive: jest.fn(() => Promise.resolve()),

    // For NativeEventEmitter
    addListener: jest.fn(),
    removeListeners: jest.fn(),
  }
}

module.exports = jest.requireActual('whisper.rn/index')
