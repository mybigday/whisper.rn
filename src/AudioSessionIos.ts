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

export enum AudioSessionCategoryOption {
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

/**
 * AudioSession Utility, iOS only.
 */
export default {
  Category: AudioSessionCategory,
  CategoryOption: AudioSessionCategoryOption,
  Mode: AudioSessionMode,

  getCurrentCategory: async (): Promise<{
    category: AudioSessionCategory,
    options: AudioSessionCategoryOption[],
  }> => {
    checkPlatform()
    const result = await RNWhisper.getAudioSessionCurrentCategory()
    return {
      category: (result.category.replace('AVAudioSessionCategory', '') as AudioSessionCategory),
      options: result.options?.map((option: string) => (option.replace('AVAudioSessionCategoryOption', '') as AudioSessionCategoryOption)),
    }
  },
  
  getCurrentMode: async (): Promise<AudioSessionMode> => {
    checkPlatform()
    const mode = await RNWhisper.getAudioSessionCurrentMode()
    return (mode.replace('AVAudioSessionMode', '') as AudioSessionMode)
  },

  setCategory: async (
    category: AudioSessionCategory,
    options: AudioSessionCategoryOption[],
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
