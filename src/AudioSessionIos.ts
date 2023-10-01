import { Platform } from 'react-native'
import RNWhisper from './NativeRNWhisper'

export enum AudioSessionCategory {
  Ambient = 'Ambient',
  SoloAmbient = 'SoloAmbient',
  Playback = 'Playback',
  Record = 'Record',
  PlayAndRecord = 'PlayAndRecord',
  MultiRoute = 'MultiRoute',
}

export enum AudioSessionCategoryOptions {
  MixWithOthers = 'MixWithOthers',
  DuckOthers = 'DuckOthers',
  InterruptSpokenAudioAndMixWithOthers = 'InterruptSpokenAudioAndMixWithOthers',
  AllowBluetooth = 'AllowBluetooth',
  AllowBluetoothA2DP = 'AllowBluetoothA2DP',
  AllowAirPlay = 'AllowAirPlay',
  DefaultToSpeaker = 'DefaultToSpeaker',
}

export enum AudioSessionMode {
  Default = 'Default',
  VoiceChat = 'VoiceChat',
  VideoChat = 'VideoChat',
  GameChat = 'GameChat',
  VideoRecording = 'VideoRecording',
  Measurement = 'Measurement',
  MoviePlayback = 'MoviePlayback',
  SpokenAudio = 'SpokenAudio',
}

const checkPlatform = () => {
  if (Platform.OS !== 'ios') throw new Error('Only supported on iOS')
}

export default {
  getCurrentCategory: async (): Promise<{
    category: AudioSessionCategory,
    options: AudioSessionCategoryOptions[],
  }> => {
    checkPlatform()
    const { category, options } = await RNWhisper.getAudioSessionCurrentCategory()
    return {
      category: category as AudioSessionCategory,
      options: options as AudioSessionCategoryOptions[],
    }
  },
  
  getCurrentMode: async (): Promise<AudioSessionMode> => {
    checkPlatform()
    const mode = await RNWhisper.getAudioSessionCurrentMode()
    return mode as AudioSessionMode
  },

  setCategory: async (
    category: AudioSessionCategory,
    options: AudioSessionCategoryOptions[],
  ): Promise<void> => {
    checkPlatform()
    await RNWhisper.setAudioSessionCategory(category, options)
  },

  setMode: async (mode: AudioSessionMode): Promise<void> => {
    checkPlatform()
    await RNWhisper.setAudioSessionMode(mode)
  },

  setActive: async (active: boolean): Promise<void> => {
    checkPlatform()
    await RNWhisper.setAudioSessionActive(active)
  },
}
