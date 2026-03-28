// Main transcriber class
export { RealtimeTranscriber } from './RealtimeTranscriber'

// Slice manager (for advanced use cases)
export { SliceManager } from './SliceManager'

// Ring buffer for pre-recording audio
export { RingBuffer } from './RingBuffer'
export { RingBufferVad } from './RingBufferVad'

export type { WavFileWriterFs } from '../utils/WavFileWriter'

// Types and interfaces
export type {
  // Audio Stream types
  AudioStreamData,
  AudioStreamConfig,
  AudioStreamInterface,

  // VAD and event types
  RealtimeVadEvent,
  RealtimeTranscribeEvent,
  RealtimeStatsEvent,

  // Configuration types
  RealtimeTranscriberDependencies,
  RealtimeOptions,
  RealtimeTranscriberCallbacks,

  // Audio slice types
  AudioSlice,
  AudioSliceNoData,
  MemoryUsage,
} from './types'

// VAD presets constant
export { VAD_PRESETS } from './types'
