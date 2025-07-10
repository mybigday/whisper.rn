import { Platform } from 'react-native'
import RNWhisper from './NativeRNWhisper'

/**
 * @see https://developer.apple.com/documentation/avfaudio/avaudiosessioncategory?language=objc
 */
export enum AudioSessionCategoryIos {
  Ambient = 'Ambient',
  SoloAmbient = 'SoloAmbient',
  Playback = 'Playback',
  Record = 'Record',
  PlayAndRecord = 'PlayAndRecord',
  MultiRoute = 'MultiRoute',
}

/**
 * @see https://developer.apple.com/documentation/avfaudio/avaudiosessioncategoryoptions?language=objc
 */
export enum AudioSessionCategoryOptionIos {
  MixWithOthers = 'MixWithOthers',
  DuckOthers = 'DuckOthers',
  InterruptSpokenAudioAndMixWithOthers = 'InterruptSpokenAudioAndMixWithOthers',
  AllowBluetooth = 'AllowBluetooth',
  AllowBluetoothA2DP = 'AllowBluetoothA2DP',
  AllowAirPlay = 'AllowAirPlay',
  DefaultToSpeaker = 'DefaultToSpeaker',
}

/**
 * @see https://developer.apple.com/documentation/avfaudio/avaudiosessionmode?language=objc
 */
export enum AudioSessionModeIos {
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
  console.warn('AudioSessionIos is deprecated. To use whisper.rn for realtime transcription, use the new RealtimeTranscriber instead.')
}

/**
 * [Deprecated] AudioSession Utility, iOS only.
 */
export default {
  Category: AudioSessionCategoryIos,
  CategoryOption: AudioSessionCategoryOptionIos,
  Mode: AudioSessionModeIos,

  getCurrentCategory: async (): Promise<{
    category: AudioSessionCategoryIos,
    options: AudioSessionCategoryOptionIos[],
  }> => {
    checkPlatform()
    const result = await RNWhisper.getAudioSessionCurrentCategory()
    return {
      category: (result.category.replace('AVAudioSessionCategory', '') as AudioSessionCategoryIos),
      options: result.options?.map((option: string) => (option.replace('AVAudioSessionCategoryOption', '') as AudioSessionCategoryOptionIos)),
    }
  },

  getCurrentMode: async (): Promise<AudioSessionModeIos> => {
    checkPlatform()
    const mode = await RNWhisper.getAudioSessionCurrentMode()
    return (mode.replace('AVAudioSessionMode', '') as AudioSessionModeIos)
  },

  setCategory: async (
    category: AudioSessionCategoryIos,
    options: AudioSessionCategoryOptionIos[],
  ): Promise<void> => {
    checkPlatform()
    await RNWhisper.setAudioSessionCategory(category, options)
  },

  setMode: async (mode: AudioSessionModeIos): Promise<void> => {
    checkPlatform()
    await RNWhisper.setAudioSessionMode(mode)
  },

  setActive: async (active: boolean): Promise<void> => {
    checkPlatform()
    await RNWhisper.setAudioSessionActive(active)
  },
}
