import type {
  TranscribeFileOptions,
  TranscribeResult,
  VadOptions,
} from '../../../src'

// === Audio Stream Interfaces ===

export interface AudioStreamData {
  data: Uint8Array
  sampleRate: number
  channels: number
  timestamp: number
}

export interface AudioStreamConfig {
  sampleRate: number
  channels: number
  bitsPerSample: number
  bufferSize?: number
  audioSource?: number
}

export interface AudioStreamInterface {
  initialize(config: AudioStreamConfig): Promise<void>
  start(): Promise<void>
  stop(): Promise<void>
  isRecording(): boolean
  onData(callback: (data: AudioStreamData) => void): void
  onError(callback: (error: string) => void): void
  onStatusChange(callback: (isRecording: boolean) => void): void
  release(): Promise<void>
}

// === Enhanced VAD Options ===

// Pre-defined VAD configurations for different use cases
export const VAD_PRESETS = {
  // Default - balanced performance
  DEFAULT: {
    threshold: 0.5,
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100,
    maxSpeechDurationS: 30,
    speechPadMs: 30,
    samplesOverlap: 0.1,
  },

  // Sensitive - good for quiet environments
  SENSITIVE: {
    threshold: 0.3,
    minSpeechDurationMs: 100,
    minSilenceDurationMs: 50,
    maxSpeechDurationS: 15,
    speechPadMs: 50,
    samplesOverlap: 0.2,
  },

  // Very sensitive - catches even quiet speech
  VERY_SENSITIVE: {
    threshold: 0.2,
    minSpeechDurationMs: 100,
    minSilenceDurationMs: 50,
    maxSpeechDurationS: 15,
    speechPadMs: 100,
    samplesOverlap: 0.3,
  },

  // Conservative - avoids false positives
  CONSERVATIVE: {
    threshold: 0.7,
    minSpeechDurationMs: 500,
    minSilenceDurationMs: 200,
    maxSpeechDurationS: 25,
    speechPadMs: 20,
    samplesOverlap: 0.05,
  },

  // Very conservative - only clear speech
  VERY_CONSERVATIVE: {
    threshold: 0.8,
    minSpeechDurationMs: 750,
    minSilenceDurationMs: 300,
    maxSpeechDurationS: 20,
    speechPadMs: 10,
    samplesOverlap: 0.05,
  },

  // Continuous speech - for presentations/lectures
  CONTINUOUS_SPEECH: {
    threshold: 0.4,
    minSpeechDurationMs: 200,
    minSilenceDurationMs: 300,
    maxSpeechDurationS: 60, // Longer segments
    speechPadMs: 50,
    samplesOverlap: 0.15,
  },

  // Meeting mode - handles multiple speakers
  MEETING: {
    threshold: 0.45,
    minSpeechDurationMs: 300,
    minSilenceDurationMs: 150,
    maxSpeechDurationS: 45,
    speechPadMs: 75,
    samplesOverlap: 0.2,
  },

  // Noisy environment - more strict thresholds
  NOISY_ENVIRONMENT: {
    threshold: 0.75,
    minSpeechDurationMs: 400,
    minSilenceDurationMs: 100,
    maxSpeechDurationS: 25,
    speechPadMs: 40,
    samplesOverlap: 0.1,
  },
}

export interface VADEvent {
  type: 'speech_start' | 'speech_end' | 'speech_continue' | 'silence'
  timestamp: number
  lastSpeechDetectedTime: number
  confidence: number
  duration: number
  sliceIndex: number

  // Additional context
  analysis?: {
    averageAmplitude: number
    peakAmplitude: number
    spectralCentroid?: number
    zeroCrossingRate?: number
  }

  // Adaptive threshold info
  currentThreshold?: number
  environmentNoise?: number
}

export interface TranscribeEvent {
  type: 'start' | 'transcribe' | 'end' | 'error'
  sliceIndex: number
  data?: TranscribeResult
  isCapturing: boolean
  processTime: number
  recordingTime: number
  memoryUsage?: {
    slicesInMemory: number
    totalSamples: number
    estimatedMB: number
  }
  vadEvent?: VADEvent
}

export interface RealtimeOptions {
  // Audio settings
  audioSliceSec?: number // default: 25
  audioMinSec?: number // default: 1
  maxSlicesInMemory?: number // default: 3

  // VAD settings - now using extended options
  vadOptions?: VadOptions
  vadPreset?: keyof typeof VAD_PRESETS // Quick preset selection

  // Auto-slice settings
  autoSliceOnSpeechEnd?: boolean // default: false - automatically slice when speech ends and duration thresholds are met
  autoSliceThreshold?: number // default: 0.85 - percentage of audioSliceSec to trigger auto-slice

  // Transcription settings
  transcribeOptions?: TranscribeFileOptions

  // Prompt settings
  initialPrompt?: string // Initial prompt to use for transcription
  promptPreviousSlices?: boolean // Add transcription results from previous slices as prompt (default: true)

  // File settings
  audioOutputPath?: string

  // Audio stream configuration
  audioStreamConfig?: AudioStreamConfig
}

export interface AudioSlice {
  index: number
  data: Uint8Array
  sampleCount: number
  startTime: number
  endTime: number
  isProcessed: boolean
  isReleased: boolean
}

export interface AudioSliceNoData extends Omit<AudioSlice, 'data'> {}

export interface MemoryUsage {
  slicesInMemory: number
  totalSamples: number
  estimatedMB: number
}

export interface StatsEvent {
  timestamp: number
  type:
    | 'slice_processed'
    | 'vad_change'
    | 'queue_change'
    | 'memory_change'
    | 'status_change'
  data: {
    isActive: boolean
    isTranscribing: boolean
    vadEnabled: boolean
    queueLength: number
    audioStats: any
    vadStats: any
    sliceStats: any
  }
}

export interface RealtimeTranscriberCallbacks {
  onTranscribe?: (event: TranscribeEvent) => void
  onVAD?: (event: VADEvent) => void
  onError?: (error: string) => void
  onStatusChange?: (isActive: boolean) => void
  onStatsUpdate?: (event: StatsEvent) => void
}

// === Context Interfaces ===

export interface RealtimeTranscriberContexts {
  whisperContext: import('../../../src').WhisperContext
  vadContext?: import('../../../src').WhisperVadContext
}

export interface RealtimeTranscriberDependencies {
  contexts: RealtimeTranscriberContexts
  audioStream: AudioStreamInterface
}
