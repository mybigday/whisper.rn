import type { AudioStreamInterface, AudioStreamConfig, AudioStreamData } from '../types'

export interface JestAudioStreamAdapterOptions {
  sampleRate?: number
  channels?: number
  bitsPerSample?: number
  simulateLatency?: number // milliseconds
  simulateErrors?: boolean
  simulateStartErrorOnly?: boolean // only simulate errors on start, not initialize
  chunkSize?: number // bytes per chunk
  chunkInterval?: number // milliseconds between chunks
  maxChunks?: number // maximum number of chunks to send
  audioData?: Uint8Array // pre-defined audio data to stream
  generateSilence?: boolean // generate silence if no audioData provided
}

export class JestAudioStreamAdapter implements AudioStreamInterface {
  private config: AudioStreamConfig | null = null

  private options: JestAudioStreamAdapterOptions

  private isInitialized = false

  private recording = false

  private dataCallback?: (data: AudioStreamData) => void

  private errorCallback?: (error: string) => void

  private statusCallback?: (isRecording: boolean) => void

  private streamInterval?: ReturnType<typeof setTimeout>

  private chunksSent = 0

  private startTime = 0

  constructor(options: JestAudioStreamAdapterOptions = {}) {
    this.options = {
      sampleRate: 16000,
      channels: 1,
      bitsPerSample: 16,
      simulateLatency: 0,
      simulateErrors: false,
      chunkSize: 3200, // 100ms at 16kHz, 16-bit, mono
      chunkInterval: 100, // 100ms
      maxChunks: -1, // unlimited
      generateSilence: true,
      ...options,
    }
  }

  async initialize(config: AudioStreamConfig): Promise<void> {
    if (this.isInitialized) {
      await this.release()
    }

    if (this.options.simulateLatency! > 0) {
      await JestAudioStreamAdapter.delay(this.options.simulateLatency!)
    }

    if (this.options.simulateErrors && !this.options.simulateStartErrorOnly) {
      throw new Error('Simulated initialization error')
    }

    this.config = config
    this.isInitialized = true
  }

  async start(): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('AudioStream not initialized')
    }

    if (this.recording) {
      return
    }

    if (this.options.simulateLatency! > 0) {
      await JestAudioStreamAdapter.delay(this.options.simulateLatency!)
    }

    if (this.options.simulateErrors) {
      throw new Error('Simulated start error')
    }

    this.recording = true
    this.chunksSent = 0
    this.startTime = Date.now()
    this.statusCallback?.(true)
    this.startStreaming()
  }

  async stop(): Promise<void> {
    if (!this.recording) {
      return
    }

    if (this.options.simulateLatency! > 0) {
      await JestAudioStreamAdapter.delay(this.options.simulateLatency!)
    }

    this.recording = false
    this.statusCallback?.(false)

    if (this.streamInterval) {
      clearTimeout(this.streamInterval)
      this.streamInterval = undefined
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
    if (this.recording) {
      await this.stop()
    }

    this.isInitialized = false
    this.config = null
    this.dataCallback = undefined
    this.errorCallback = undefined
    this.statusCallback = undefined
    this.chunksSent = 0
  }

  // Test helper methods
  simulateError(error: string): void {
    this.errorCallback?.(error)
  }

  simulateDataChunk(data: Uint8Array): void {
    if (!this.dataCallback || !this.config) {
      return
    }

    const streamData: AudioStreamData = {
      data,
      sampleRate: this.config.sampleRate || this.options.sampleRate!,
      channels: this.config.channels || this.options.channels!,
      timestamp: Date.now(),
    }

    this.dataCallback(streamData)
  }

  getChunksSent(): number {
    return this.chunksSent
  }

  getTotalBytesStreamed(): number {
    return this.chunksSent * this.options.chunkSize!
  }

  getStreamDuration(): number {
    return this.recording ? Date.now() - this.startTime : 0
  }

    private startStreaming(): void {
    if (!this.dataCallback || !this.config) {
      return
    }

    const streamChunk = () => {
      if (!this.recording) {
        return
      }

      // Check if we've reached the maximum chunks
      if (this.options.maxChunks! > 0 && this.chunksSent >= this.options.maxChunks!) {
        this.stop()
        return
      }

      // Generate or use provided audio data
      const audioData = this.generateAudioChunk()

      if (audioData) {
        this.simulateDataChunk(audioData)
        this.chunksSent += 1
      }

      // Schedule next chunk if still recording
      if (this.recording) {
        this.streamInterval = setTimeout(streamChunk, this.options.chunkInterval!)
      }
    }

    // Start streaming after a short delay
    this.streamInterval = setTimeout(streamChunk, this.options.chunkInterval!)
  }

  private generateAudioChunk(): Uint8Array | null {
    // If we have pre-defined audio data, use it
    if (this.options.audioData) {
      const startByte = this.chunksSent * this.options.chunkSize!
      const endByte = Math.min(startByte + this.options.chunkSize!, this.options.audioData.length)

      if (startByte >= this.options.audioData.length) {
        return null // No more data
      }

      return this.options.audioData.subarray(startByte, endByte)
    }

    // Generate silence or simple tone
    const chunkSize = this.options.chunkSize!
    const audioData = new Uint8Array(chunkSize)

    if (this.options.generateSilence) {
      // Generate silence (all zeros)
      audioData.fill(0)
    } else {
      // Generate a simple sine wave tone for testing
      const sampleRate = this.options.sampleRate!
      const frequency = 440 // A4 note
      const samplesPerChunk = chunkSize / 2 // 16-bit samples
      const timeOffset = (this.chunksSent * samplesPerChunk) / sampleRate

            for (let i = 0; i < samplesPerChunk; i += 1) {
        const time = timeOffset + i / sampleRate
        const amplitude = Math.sin(2 * Math.PI * frequency * time) * 0.5
        const sample = Math.round(amplitude * 32767) // 16-bit signed sample

        // Convert to little-endian bytes
        audioData[i * 2] = sample % 256
        audioData[i * 2 + 1] = Math.floor(sample / 256) % 256
      }
    }

    return audioData
  }

  private static delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
}
