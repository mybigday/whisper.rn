// Main transcriber class
export { RealtimeTranscriber } from './RealtimeTranscriber'

// Slice manager (for advanced use cases)
export { SliceManager } from './SliceManager'

// Ring buffer for pre-recording audio
export { RingBuffer } from './RingBuffer'
export { RingBufferVad } from './RingBufferVad'

// Stateful VAD (vad.cpp streams)
export { StreamVad } from './StreamVad'
export type { StreamVadOptions, StreamVadContextLike, VadStreamLike } from './StreamVad'
export { StreamingVadSegmenter } from './StreamingVadSegmenter'
export type { VadSegmenterEvent } from './StreamingVadSegmenter'

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
