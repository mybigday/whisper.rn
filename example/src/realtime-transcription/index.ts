// Main transcriber class
export { RealtimeTranscriber } from './RealtimeTranscriber'

// Audio stream adapters
export { LiveAudioStreamAdapter } from './LiveAudioStreamAdapter'
export { SimulateFileAudioStreamAdapter } from './SimulateFileAudioStreamAdapter'

// Slice manager (for advanced use cases)
export { SliceManager } from './SliceManager'

// Types and interfaces
export type {
  // Audio Stream types
  AudioStreamData,
  AudioStreamConfig,
  AudioStreamInterface,

  // VAD and event types
  VADEvent,
  TranscribeEvent,
  StatsEvent,

  // Configuration types
  RealtimeOptions,
  RealtimeTranscriberCallbacks,
  RealtimeTranscriberContexts,
  RealtimeTranscriberDependencies,

  // Audio slice types
  AudioSlice,
  AudioSliceNoData,
  MemoryUsage,
} from './types'

// VAD presets constant
export { VAD_PRESETS } from './types'
