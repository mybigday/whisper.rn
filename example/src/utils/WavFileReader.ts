import RNFS from 'react-native-fs'

export interface WavFileHeader {
  sampleRate: number
  channels: number
  bitsPerSample: number
  dataSize: number
  duration: number
}

export class WavFileReader {
  private filePath: string

  private header: WavFileHeader | null = null

  private audioData: Uint8Array | null = null

  constructor(filePath: string) {
    this.filePath = filePath
  }

  /**
   * Read and parse the WAV file
   */
  async initialize(): Promise<void> {
    try {
      // Check if file exists
      const exists = await RNFS.exists(this.filePath)
      if (!exists) {
        throw new Error(`WAV file not found: ${this.filePath}`)
      }

      // Read the entire file
      const fileContent = await RNFS.readFile(this.filePath, 'base64')
      const fileData = WavFileReader.base64ToUint8Array(fileContent)

      // Parse WAV header
      this.header = WavFileReader.parseWavHeader(fileData)

      // Extract audio data (skip the 44-byte header)
      this.audioData = fileData.slice(44, 44 + this.header.dataSize)

      console.log(
        `WAV file loaded: ${this.header.duration.toFixed(2)}s, ${
          this.header.sampleRate
        }Hz, ${this.header.channels}ch`,
      )
    } catch (error) {
      throw new Error(`Failed to initialize WAV file reader: ${error}`)
    }
  }

  /**
   * Parse WAV file header
   */
  private static parseWavHeader(data: Uint8Array): WavFileHeader {
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength)

    // Verify RIFF header
    const riffHeader = String.fromCharCode(...data.slice(0, 4))
    if (riffHeader !== 'RIFF') {
      throw new Error('Invalid WAV file: Missing RIFF header')
    }

    // Verify WAVE format
    const waveHeader = String.fromCharCode(...data.slice(8, 12))
    if (waveHeader !== 'WAVE') {
      throw new Error('Invalid WAV file: Missing WAVE header')
    }

    // Read format chunk
    const fmtHeader = String.fromCharCode(...data.slice(12, 16))
    if (fmtHeader !== 'fmt ') {
      throw new Error('Invalid WAV file: Missing fmt chunk')
    }

    const audioFormat = view.getUint16(20, true)
    if (audioFormat !== 1) {
      throw new Error('Unsupported WAV format: Only PCM is supported')
    }

    const channels = view.getUint16(22, true)
    const sampleRate = view.getUint32(24, true)
    const bitsPerSample = view.getUint16(34, true)

    // Find data chunk
    let dataOffset = 36
    while (dataOffset < data.length - 8) {
      const chunkId = String.fromCharCode(
        ...data.slice(dataOffset, dataOffset + 4),
      )
      const chunkSize = view.getUint32(dataOffset + 4, true)

      if (chunkId === 'data') {
        const dataSize = chunkSize
        const duration =
          dataSize / (sampleRate * channels * (bitsPerSample / 8))

        return {
          sampleRate,
          channels,
          bitsPerSample,
          dataSize,
          duration,
        }
      }

      dataOffset += 8 + chunkSize
    }

    throw new Error('Invalid WAV file: Missing data chunk')
  }

  /**
   * Get audio data slice
   */
  getAudioSlice(startByte: number, lengthBytes: number): Uint8Array | null {
    if (!this.audioData) {
      return null
    }

    const start = Math.max(0, startByte)
    const end = Math.min(this.audioData.length, startByte + lengthBytes)

    if (start >= end) {
      return null
    }

    return this.audioData.slice(start, end)
  }

  getAudioData(): Uint8Array | null {
    return this.audioData
  }

  getAudioDataBase64(): string | null {
    if (!this.audioData) {
      return null
    }
    return WavFileReader.uint8ArrayToBase64(this.audioData)
  }

  /**
   * Get WAV file header information
   */
  getHeader(): WavFileHeader | null {
    return this.header
  }

  /**
   * Get total audio data size
   */
  getTotalDataSize(): number {
    return this.header?.dataSize || 0
  }

  /**
   * Convert byte position to time in seconds
   */
  byteToTime(bytePosition: number): number {
    if (!this.header) return 0

    const bytesPerSecond =
      this.header.sampleRate *
      this.header.channels *
      (this.header.bitsPerSample / 8)
    return bytePosition / bytesPerSecond
  }

  /**
   * Convert time in seconds to byte position
   */
  timeToByte(timeSeconds: number): number {
    if (!this.header) return 0

    const bytesPerSecond =
      this.header.sampleRate *
      this.header.channels *
      (this.header.bitsPerSample / 8)
    return Math.floor(timeSeconds * bytesPerSecond)
  }

  /**
   * Convert base64 string to Uint8Array
   */
  private static base64ToUint8Array(base64: string): Uint8Array {
    const binaryString = atob(base64)
    const bytes = new Uint8Array(binaryString.length)
    for (let i = 0; i < binaryString.length; i += 1) {
      bytes[i] = binaryString.charCodeAt(i)
    }
    return bytes
  }

  private static uint8ArrayToBase64(buffer: Uint8Array): string {
    let binary = ''
    for (let i = 0; i < buffer.length; i += 1) {
      binary += String.fromCharCode(buffer[i] || 0) // Handle undefined
    }
    return btoa(binary)
  }

  /**
   * Get file statistics
   */
  getStatistics() {
    return {
      filePath: this.filePath,
      header: this.header,
      totalDataSize: this.getTotalDataSize(),
      isInitialized: !!this.header,
    }
  }
}
