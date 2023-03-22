const { NativeModules } = require('react-native')

if (!NativeModules.RNWhisper) {
  NativeModules.RNWhisper = {
    initContext: jest.fn(() => Promise.resolve(1)),
    transcribe: jest.fn(() => Promise.resolve({
      result: ' Test',
      segments: [{ text: ' Test', t0: 0, t1: 33 }],
    })),
    releaseContext: jest.fn(() => Promise.resolve()),
    releaseAllContexts: jest.fn(() => Promise.resolve()),
  }
}

module.exports = jest.requireActual('whisper.rn')
