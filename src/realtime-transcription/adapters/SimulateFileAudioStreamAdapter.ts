import type {
  AudioStreamInterface,
  AudioStreamConfig,
  AudioStreamData,
} from '../types'
import { WavFileReader, WavFileReaderFs } from '../../utils/WavFileReader'

export interface SimulateFileOptions {
  fs: WavFileReaderFs
  filePath: string
  playbackSpeed?: number // Default: 1.0 (real-time), 0.5 (half speed), 2.0 (double speed)
  chunkDurationMs?: number // Default: 100ms chunks
  loop?: boolean // Default: false
  onEndOfFile?: () => void // Callback when end of file is reached
  logger?: (message: string) => void // Default: noop - custom logger function
}

export class SimulateFileAudioStreamAdapter implements AudioStreamInterface {
  private fileReader: WavFileReader

  private config: AudioStreamConfig | null = null

  private options: SimulateFileOptions

  private isInitialized = false

  private recording = false

  private dataCallback?: (data: AudioStreamData) => void

  private errorCallback?: (error: string) => void

  private statusCallback?: (isRecording: boolean) => void

  private streamInterval?: ReturnType<typeof setInterval>

  private currentBytePosition = 0

  private startTime = 0

  private pausedTime = 0

  private hasReachedEnd = false

  constructor(options: SimulateFileOptions) {
    this.options = {
      playbackSpeed: 1.0,
      chunkDurationMs: 100,
      loop: false,
      logger: () => {},
      ...options,
    }
    this.fileReader = new WavFileReader(this.options.fs, this.options.filePath)
  }

  async initialize(config: AudioStreamConfig): Promise<void> {
    if (this.isInitialized) {
      await this.release()
    }

    try {
      this.config = config

      // Initialize the WAV file reader
      await this.fileReader.initialize()

      // Validate file format matches config
      const header = this.fileReader.getHeader()
      if (!header) {
        throw new Error('Failed to read WAV file header')
      }

      // Warn about mismatched formats but allow processing
      if (header.sampleRate !== config.sampleRate) {
        this.log(
          `WAV file sample rate (${header.sampleRate}Hz) differs from config (${config.sampleRate}Hz)`,
        )
      }

      if (header.channels !== config.channels) {
        this.log(
          `WAV file channels (${header.channels}) differs from config (${config.channels})`,
        )
      }

      if (header.bitsPerSample !== config.bitsPerSample) {
        this.log(
          `WAV file bits per sample (${header.bitsPerSample}) differs from config (${config.bitsPerSample})`,
        )
      }

      this.isInitialized = true
      this.log(
        `Simulate audio stream initialized: ${header.duration.toFixed(2)}s at ${
          this.options.playbackSpeed
        }x speed`,
      )
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown initialization error'
      this.errorCallback?.(errorMessage)
      throw new Error(
        `Failed to initialize SimulateFileAudioStreamAdapter: ${errorMessage}`,
      )
    }
  }

  async start(): Promise<void> {
    if (!this.isInitialized || !this.config) {
      throw new Error('Adapter not initialized')
    }

    if (this.recording) {
      return
    }

    try {
      this.recording = true
      this.hasReachedEnd = false
      this.startTime = Date.now() - this.pausedTime
      this.statusCallback?.(true)

      // Start streaming chunks
      this.startStreaming()

      this.log('File audio simulation started')
    } catch (error) {
      this.recording = false
      this.statusCallback?.(false)
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown start error'
      this.errorCallback?.(errorMessage)
      throw error
    }
  }

  async stop(): Promise<void> {
    if (!this.recording) {
      return
    }

    try {
      this.recording = false
      this.pausedTime = Date.now() - this.startTime

      // Stop the streaming interval
      if (this.streamInterval) {
        clearInterval(this.streamInterval)
        this.streamInterval = undefined
      }

      this.statusCallback?.(false)
      this.log('File audio simulation stopped')
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown stop error'
      this.errorCallback?.(errorMessage)
    }
  }

  isRecording(): boolean {
    return this.recording
  }

  onData(callback: (data: AudioStreamData) => void): void {
    this.dataCallback = callback
  }

  onError(callback: (error: string) => void): void {
    this.errorCallback = callback
  }

  onStatusChange(callback: (isRecording: boolean) => void): void {
    this.statusCallback = callback
  }

  async release(): Promise<void> {
    await this.stop()
    this.isInitialized = false
    this.currentBytePosition = 0
    this.pausedTime = 0
    this.log('SimulateFileAudioStreamAdapter released')
  }

  /**
   * Start the streaming process
   */
  private startStreaming(): void {
    if (!this.config || !this.isInitialized) {
      return
    }

    const header = this.fileReader.getHeader()
    if (!header) {
      this.errorCallback?.('WAV file header not available')
      return
    }

    // Calculate chunk size based on desired duration
    const chunkDurationSec = (this.options.chunkDurationMs || 100) / 1000
    const bytesPerSecond =
      header.sampleRate * header.channels * (header.bitsPerSample / 8)
    const chunkSizeBytes = Math.floor(chunkDurationSec * bytesPerSecond)

    // Adjust interval timing based on playback speed
    const intervalMs =
      (this.options.chunkDurationMs || 100) /
      (this.options.playbackSpeed || 1.0)

    this.streamInterval = setInterval(() => {
      if (!this.recording) {
        return
      }

      try {
        this.streamNextChunk(chunkSizeBytes)
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Streaming error'
        this.errorCallback?.(errorMessage)
        this.stop()
      }
    }, intervalMs)
  }

  /**
   * Stream the next audio chunk
   */
  private streamNextChunk(chunkSizeBytes: number): void {
    if (!this.dataCallback || !this.config) {
      return
    }

    const header = this.fileReader.getHeader()
    if (!header) {
      return
    }

    // Get the next chunk of audio data
    const audioChunk = this.fileReader.getAudioSlice(
      this.currentBytePosition,
      chunkSizeBytes,
    )

    if (!audioChunk || audioChunk.length === 0) {
      // End of file reached
      if (this.options.loop) {
        // Reset to beginning for looping
        this.currentBytePosition = 0
        this.startTime = Date.now()
        this.pausedTime = 0
        this.hasReachedEnd = false
        this.log('Looping audio file simulation')
        return
      }

      // Stop streaming due to no new buffer
      this.log('Audio file simulation completed - no new buffer available')
      this.hasReachedEnd = true

      // Call the end-of-file callback if provided
      if (this.options.onEndOfFile) {
        this.log('Calling onEndOfFile callback')
        this.options.onEndOfFile()
      }

      // Stop the stream
      this.stop()
      return
    }

    // Update position
    this.currentBytePosition += audioChunk.length

    // Create stream data using the original file's format
    const streamData: AudioStreamData = {
      data: audioChunk,
      sampleRate: header.sampleRate,
      channels: header.channels,
      timestamp: Date.now(),
    }

    // Send the chunk
    this.dataCallback(streamData)
  }

  /**
   * Get current playback statistics
   */
  getStatistics() {
    const header = this.fileReader.getHeader()
    const currentTime = this.fileReader.byteToTime(this.currentBytePosition)

    return {
      filePath: this.options.filePath,
      isRecording: this.recording,
      currentTime,
      totalDuration: header?.duration || 0,
      progress: header ? currentTime / header.duration : 0,
      playbackSpeed: this.options.playbackSpeed,
      currentBytePosition: this.currentBytePosition,
      totalBytes: this.fileReader.getTotalDataSize(),
      hasReachedEnd: this.hasReachedEnd,
      header,
    }
  }

  /**
   * Seek to a specific time position
   */
  seekToTime(timeSeconds: number): void {
    const header = this.fileReader.getHeader()
    if (!header) {
      return
    }

    const clampedTime = Math.max(0, Math.min(timeSeconds, header.duration))
    this.currentBytePosition = this.fileReader.timeToByte(clampedTime)

    // Reset timing if we're currently playing
    if (this.recording) {
      this.startTime =
        Date.now() - (clampedTime * 1000) / (this.options.playbackSpeed || 1.0)
      this.pausedTime = 0
    }

    this.log(`Seeked to ${clampedTime.toFixed(2)}s`)
  }

  /**
   * Set playback speed
   */
  setPlaybackSpeed(speed: number): void {
    if (speed <= 0) {
      throw new Error('Playback speed must be greater than 0')
    }

    this.options.playbackSpeed = speed

    // If currently playing, restart streaming with new speed
    if (this.recording) {
      this.stop().then(() => {
        this.start()
      })
    }

    this.log(`Playback speed set to ${speed}x`)
  }

  /**
   * Reset file buffer to beginning
   */
  resetBuffer(): void {
    this.log('Resetting file buffer to beginning')

    // Reset position and timing
    this.currentBytePosition = 0
    this.startTime = Date.now()
    this.pausedTime = 0
    this.hasReachedEnd = false

    // If currently playing, restart streaming from beginning
    if (this.recording) {
      this.log('Restarting streaming from beginning')
      // Stop and restart to apply the reset
      this.stop().then(() => {
        this.start()
      })
    }
  }

  /**
   * Logger function
   */
  private log(message: string): void {
    this.options.logger?.(`[SimulateFileAudioStreamAdapter] ${message}`)
  }
}
