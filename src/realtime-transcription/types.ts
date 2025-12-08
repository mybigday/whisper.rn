import type { TranscribeOptions, TranscribeResult, VadOptions } from '../index'
import type { WavFileWriterFs } from '../utils/WavFileWriter'

// === Audio Stream Interfaces ===

export interface AudioStreamData {
  data: Uint8Array
  sampleRate: number
  channels: number
  timestamp: number
}

export interface AudioStreamConfig {
  sampleRate?: number
  channels?: number
  bitsPerSample?: number
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
/**
 * VAD Presets Overview:
 *
 *                            VAD Presets
 *                         /      |      \
 *                Conservative  Default  Sensitive
 *                /        |        |        \
 *        conservative  very-conservative  sensitive  very-sensitive
 *        (0.7 thresh)   (0.8 thresh)    (0.3 thresh) (0.2 thresh)
 *        500ms min      750ms min       100ms min    100ms min
 *        Clear speech   Very clear      Quiet env    Catches whispers
 *
 *                         Specialized Presets
 *                      /        |        \
 *                continuous   meeting    noisy
 *                (60s max)    (45s max)  (0.75 thresh)
 *                Lectures     Multi-spk   Strict for noise
 *
 * Key Parameters:
 * - threshold: 0.0-1.0 (lower = more sensitive)
 * - minSpeechDurationMs: Min duration to consider speech
 * - minSilenceDurationMs: Min silence before ending speech
 * - maxSpeechDurationS: Max continuous speech duration
 * - speechPadMs: Padding around detected speech
 * - samplesOverlap: Analysis window overlap (0.0-1.0)
 */
export const VAD_PRESETS = {
  // Default - balanced performance
  default: {
    threshold: 0.5,
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100,
    maxSpeechDurationS: 30,
    speechPadMs: 30,
    samplesOverlap: 0.1,
  },

  // Sensitive - good for quiet environments
  sensitive: {
    threshold: 0.3,
    minSpeechDurationMs: 100,
    minSilenceDurationMs: 50,
    maxSpeechDurationS: 15,
    speechPadMs: 50,
    samplesOverlap: 0.2,
  },

  // Very sensitive - catches even quiet speech
  'very-sensitive': {
    threshold: 0.2,
    minSpeechDurationMs: 100,
    minSilenceDurationMs: 50,
    maxSpeechDurationS: 15,
    speechPadMs: 100,
    samplesOverlap: 0.3,
  },

  // Conservative - avoids false positives
  conservative: {
    threshold: 0.7,
    minSpeechDurationMs: 500,
    minSilenceDurationMs: 200,
    maxSpeechDurationS: 25,
    speechPadMs: 20,
    samplesOverlap: 0.05,
  },

  // Very conservative - only clear speech
  'very-conservative': {
    threshold: 0.8,
    minSpeechDurationMs: 750,
    minSilenceDurationMs: 300,
    maxSpeechDurationS: 20,
    speechPadMs: 10,
    samplesOverlap: 0.05,
  },

  // Continuous speech - for presentations/lectures
  continuous: {
    threshold: 0.4,
    minSpeechDurationMs: 200,
    minSilenceDurationMs: 300,
    maxSpeechDurationS: 60, // Longer segments
    speechPadMs: 50,
    samplesOverlap: 0.15,
  },

  // Meeting mode - handles multiple speakers
  meeting: {
    threshold: 0.45,
    minSpeechDurationMs: 300,
    minSilenceDurationMs: 150,
    maxSpeechDurationS: 45,
    speechPadMs: 75,
    samplesOverlap: 0.2,
  },

  // Noisy environment - more strict thresholds
  noisy: {
    threshold: 0.75,
    minSpeechDurationMs: 400,
    minSilenceDurationMs: 100,
    maxSpeechDurationS: 25,
    speechPadMs: 40,
    samplesOverlap: 0.1,
  },
}

export interface RealtimeVadEvent {
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

export interface RealtimeTranscribeEvent {
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
  vadEvent?: RealtimeVadEvent
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

  // VAD optimization options for low-end CPU
  vadThrottleMs?: number // default: 1500 - Minimum time between VAD calls (ms)
  vadSkipRatio?: number // default: 0 - Skip every Nth slice (0 = no skipping)

  // Transcription settings
  transcribeOptions?: TranscribeOptions

  // Prompt settings
  initialPrompt?: string // Initial prompt to use for transcription
  promptPreviousSlices?: boolean // Add transcription results from previous slices as prompt (default: true)

  // File settings (Only used if fs dependency is provided)
  audioOutputPath?: string

  // Audio stream configuration
  audioStreamConfig?: AudioStreamConfig

  // Logger settings
  logger?: (message: string) => void // default: noop - custom logger function
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

export interface RealtimeStatsEvent {
  timestamp: number
  type:
    | 'slice_processed'
    | 'vad_change'
    | 'memory_change'
    | 'status_change'
  data: {
    isActive: boolean
    isTranscribing: boolean
    vadEnabled: boolean
    audioStats: any
    vadStats: any
    sliceStats: any
  }
}

export interface RealtimeTranscriberCallbacks {
  onBeginTranscribe?: (sliceInfo: {
    audioData: Uint8Array
    sliceIndex: number
    duration: number
    vadEvent?: RealtimeVadEvent
  }) => Promise<boolean>
  onTranscribe?: (event: RealtimeTranscribeEvent) => void
  onBeginVad?: (sliceInfo: {
    audioData: Uint8Array
    sliceIndex: number
    duration: number
  }) => Promise<boolean>
  onVad?: (event: RealtimeVadEvent) => void
  onError?: (error: string) => void
  onStatusChange?: (isActive: boolean) => void
  onStatsUpdate?: (event: RealtimeStatsEvent) => void
}

// === Context Interfaces ===

export type WhisperContextLike = {
  transcribeData: (
    data: ArrayBuffer,
    options: TranscribeOptions,
  ) => {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  }
}

export type WhisperVadContextLike = {
  detectSpeechData: (
    data: ArrayBuffer,
    options: VadOptions,
  ) => Promise<Array<{ t0: number; t1: number }>>
}

export interface RealtimeTranscriberDependencies {
  whisperContext: WhisperContextLike
  vadContext?: WhisperVadContextLike
  audioStream: AudioStreamInterface
  fs?: WavFileWriterFs
}
