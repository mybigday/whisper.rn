import { NativeModules } from 'react-native'

if (!NativeModules.RNWhisper) {
  NativeModules.RNWhisper = {
    initContext: jest.fn(() => Promise.resolve(1)),
    transcribe: jest.fn(() => Promise.resolve('TEST')),
    releaseContext: jest.fn(() => Promise.resolve()),
    releaseAllContexts: jest.fn(() => Promise.resolve()),
  }
}

module.exports = jest.requireActual('whisper.rn')
