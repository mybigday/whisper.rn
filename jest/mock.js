const { NativeModules, DeviceEventEmitter } = require('react-native')

if (!NativeModules.RNWhisper) {
  NativeModules.RNWhisper = {
    initContext: jest.fn(() => Promise.resolve({ id: 1 })),
    transcribeFile: jest.fn(() => Promise.resolve({
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
    releaseContext: jest.fn(() => Promise.resolve()),
    releaseAllContexts: jest.fn(() => Promise.resolve()),

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

module.exports = jest.requireActual('whisper.rn')
