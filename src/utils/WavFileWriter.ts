import { base64ToUint8Array, uint8ArrayToBase64 } from './common'

export interface WavFileConfig {
  sampleRate: number
  channels: number
  bitsPerSample: number
}

export interface WavFileWriterFs {
  writeFile: (filePath: string, data: string, encoding: string) => Promise<void>
  appendFile: (filePath: string, data: string, encoding: string) => Promise<void>
  readFile: (filePath: string, encoding: string) => Promise<string>
  exists: (filePath: string) => Promise<boolean>
  unlink: (filePath: string) => Promise<void>
}

export class WavFileWriter {
  private fs: WavFileWriterFs

  private filePath: string

  private config: WavFileConfig

  private dataSize = 0

  private isWriting = false

  private writeQueue: Uint8Array[] = []

  constructor(fs: WavFileWriterFs, filePath: string, config: WavFileConfig) {
    this.fs = fs
    this.filePath = filePath
    this.config = config
  }

  /**
   * Initialize the WAV file with headers
   */
  async initialize(): Promise<void> {
    if (this.isWriting) {
      return
    }

    try {
      // Create the initial WAV header (we'll update the size later)
      const header = this.createWavHeader(0)
      await this.fs.writeFile(this.filePath, uint8ArrayToBase64(header), 'base64')

      this.dataSize = 0
      this.isWriting = true
      this.writeQueue = []
    } catch (error) {
      throw new Error(`Failed to initialize WAV file: ${error}`)
    }
  }

  /**
   * Append PCM audio data to the WAV file
   */
  async appendAudioData(audioData: Uint8Array): Promise<void> {
    if (!this.isWriting) {
      throw new Error('WAV file not initialized')
    }

    try {
      // Queue the data for writing
      this.writeQueue.push(audioData)

      // Process the write queue
      await this.processWriteQueue()
    } catch (error) {
      console.warn(`Failed to append audio data to WAV file: ${error}`)
    }
  }

  /**
   * Process the write queue to avoid blocking
   */
  private async processWriteQueue(): Promise<void> {
    if (this.writeQueue.length === 0) {
      return
    }

    try {
      // Combine all queued data
      const totalLength = this.writeQueue.reduce((sum, data) => sum + data.length, 0)
      const combinedData = new Uint8Array(totalLength)

      let offset = 0
      this.writeQueue.forEach(data => {
        combinedData.set(new Uint8Array(data), offset)
        offset += data.length
      })

      // Append to file
      const base64Data = uint8ArrayToBase64(combinedData)
      await this.fs.appendFile(this.filePath, base64Data, 'base64')

      // Update data size
      this.dataSize += combinedData.length

      // Clear the queue
      this.writeQueue = []
    } catch (error) {
      console.warn(`Failed to process WAV write queue: ${error}`)
      // Don't throw here to avoid breaking the recording
    }
  }

  /**
   * Finalize the WAV file by updating the header with correct sizes
   */
  async finalize(): Promise<void> {
    if (!this.isWriting) {
      return
    }

    try {
      // Process any remaining queued data
      await this.processWriteQueue()

      // Read the current file
      const currentData = await this.fs.readFile(this.filePath, 'base64')
      const currentBytes = base64ToUint8Array(currentData)

      // Create the correct header with final data size
      const correctHeader = this.createWavHeader(this.dataSize)

      // Replace the header (first 44 bytes)
      const finalData = new Uint8Array(correctHeader.length + this.dataSize)
      finalData.set(correctHeader, 0)
      finalData.set(currentBytes.slice(44), 44) // Skip old header

      // Write the final file
      const finalBase64 = uint8ArrayToBase64(finalData)
      await this.fs.writeFile(this.filePath, finalBase64, 'base64')

      this.isWriting = false
    } catch (error) {
      console.warn(`Failed to finalize WAV file: ${error}`)
    }
  }

  /**
   * Create WAV file header
   */
  private createWavHeader(dataSize: number): Uint8Array {
    const header = new ArrayBuffer(44)
    const view = new DataView(header)

    // RIFF header
    view.setUint32(0, 0x52494646, false) // "RIFF"
    view.setUint32(4, 36 + dataSize, true) // File size - 8
    view.setUint32(8, 0x57415645, false) // "WAVE"

    // Format chunk
    view.setUint32(12, 0x666d7420, false) // "fmt "
    view.setUint32(16, 16, true) // Chunk size
    view.setUint16(20, 1, true) // Audio format (PCM)
    view.setUint16(22, this.config.channels, true) // Number of channels
    view.setUint32(24, this.config.sampleRate, true) // Sample rate
    view.setUint32(28, this.config.sampleRate * this.config.channels * (this.config.bitsPerSample / 8), true) // Byte rate
    view.setUint16(32, this.config.channels * (this.config.bitsPerSample / 8), true) // Block align
    view.setUint16(34, this.config.bitsPerSample, true) // Bits per sample

    // Data chunk
    view.setUint32(36, 0x64617461, false) // "data"
    view.setUint32(40, dataSize, true) // Data size

    return new Uint8Array(header)
  }

  /**
   * Cancel writing and cleanup
   */
  async cancel(): Promise<void> {
    this.isWriting = false
    this.writeQueue = []

    try {
      // Delete the incomplete file
      const exists = await this.fs.exists(this.filePath)
      if (exists) {
        await this.fs.unlink(this.filePath)
      }
    } catch (error) {
      console.warn(`Failed to cleanup WAV file: ${error}`)
    }
  }

  /**
   * Get current file statistics
   */
  getStatistics() {
    const durationSec = this.dataSize / (this.config.sampleRate * this.config.channels * (this.config.bitsPerSample / 8))

    return {
      filePath: this.filePath,
      dataSize: this.dataSize,
      durationSec,
      isWriting: this.isWriting,
      queuedChunks: this.writeQueue.length,
      estimatedFileSizeMB: (44 + this.dataSize) / (1024 * 1024),
    }
  }
}
