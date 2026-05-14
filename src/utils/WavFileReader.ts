import { base64ToUint8Array } from './common'

export interface WavFileReaderFs {
  readFile: (filePath: string, encoding: string) => Promise<string>
  exists: (filePath: string) => Promise<boolean>
  unlink: (filePath: string) => Promise<void>
}

export interface WavFileHeader {
  sampleRate: number
  channels: number
  bitsPerSample: number
  dataSize: number
  duration: number
}

type ParsedWavFileHeader = {
  header: WavFileHeader
  dataOffset: number
}

export class WavFileReader {
  private filePath: string

  private header: WavFileHeader | null = null

  private audioData: Uint8Array | null = null

  private fs: {
    exists: (filePath: string) => Promise<boolean>
    readFile: (filePath: string, encoding: string) => Promise<string>
  }

  constructor(fs: {
    exists: (filePath: string) => Promise<boolean>
    readFile: (filePath: string, encoding: string) => Promise<string>
  }, filePath: string) {
    this.fs = fs
    this.filePath = filePath
  }

  /**
   * Read and parse the WAV file
   */
  async initialize(): Promise<void> {
    try {
      // Check if file exists
      const exists = await this.fs.exists(this.filePath)
      if (!exists) {
        throw new Error(`WAV file not found: ${this.filePath}`)
      }

      // Read the entire file
      const fileContent = await this.fs.readFile(this.filePath, 'base64')
      const fileData = base64ToUint8Array(fileContent)

      // Parse WAV chunks and extract audio from the actual data chunk.
      const parsedHeader = WavFileReader.parseWavHeader(fileData)
      this.header = parsedHeader.header
      this.audioData = fileData.slice(
        parsedHeader.dataOffset,
        parsedHeader.dataOffset + this.header.dataSize,
      )

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
  private static parseWavHeader(data: Uint8Array): ParsedWavFileHeader {
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength)

    // Verify RIFF header
    const riffHeader = WavFileReader.readChunkId(data, 0)
    if (riffHeader !== 'RIFF') {
      throw new Error('Invalid WAV file: Missing RIFF header')
    }

    // Verify WAVE format
    const waveHeader = WavFileReader.readChunkId(data, 8)
    if (waveHeader !== 'WAVE') {
      throw new Error('Invalid WAV file: Missing WAVE header')
    }

    let channels = 0
    let sampleRate = 0
    let bitsPerSample = 0
    let isPcm = false
    let hasFmtChunk = false
    let dataOffset = 0
    let dataSize = 0
    let offset = 12

    while (offset + 8 <= data.length) {
      const chunkId = WavFileReader.readChunkId(data, offset)
      const chunkSize = view.getUint32(offset + 4, true)
      const chunkDataOffset = offset + 8
      if (chunkDataOffset > data.length) {
        throw new Error('Invalid WAV file: Malformed chunk')
      }

      const availableBytes = data.length - chunkDataOffset
      const chunkExceedsFile = chunkSize > availableBytes
      if (chunkExceedsFile && chunkId !== 'data') {
        throw new Error('Invalid WAV file: Malformed chunk')
      }

      const effectiveChunkSize = chunkExceedsFile ? availableBytes : chunkSize
      if (chunkId === 'fmt ') {
        if (chunkSize < 16) {
          throw new Error('Invalid WAV file: Malformed fmt chunk')
        }

        const audioFormat = view.getUint16(chunkDataOffset, true)
        channels = view.getUint16(chunkDataOffset + 2, true)
        sampleRate = view.getUint32(chunkDataOffset + 4, true)
        bitsPerSample = view.getUint16(chunkDataOffset + 14, true)
        isPcm =
          audioFormat === 1 ||
          (audioFormat === 0xfffe &&
            WavFileReader.hasPcmExtensibleSubFormat(
              data,
              chunkDataOffset,
              chunkSize,
            ))
        hasFmtChunk = true
      } else if (chunkId === 'data') {
        dataOffset = chunkDataOffset
        dataSize = effectiveChunkSize
        if (hasFmtChunk) break
      }

      let nextOffset = chunkDataOffset + effectiveChunkSize
      if (!chunkExceedsFile && chunkSize % 2 !== 0 && nextOffset < data.length) {
        nextOffset += 1
      }
      if (nextOffset <= offset) {
        throw new Error('Invalid WAV file: Malformed chunk')
      }
      offset = nextOffset
    }

    if (!hasFmtChunk) {
      throw new Error('Invalid WAV file: Missing fmt chunk')
    }
    if (!dataOffset) {
      throw new Error('Invalid WAV file: Missing data chunk')
    }
    if (!isPcm) {
      throw new Error('Unsupported WAV format: Only PCM is supported')
    }
    if (!channels) {
      throw new Error('Invalid WAV file: Invalid channel count')
    }
    if (!sampleRate) {
      throw new Error('Invalid WAV file: Invalid sample rate')
    }

    const duration = dataSize / (sampleRate * channels * (bitsPerSample / 8))
    return {
      header: {
        sampleRate,
        channels,
        bitsPerSample,
        dataSize,
        duration,
      },
      dataOffset,
    }
  }

  private static readChunkId(data: Uint8Array, offset: number): string {
    if (offset + 4 > data.length) return ''
    return String.fromCharCode(
      data[offset] ?? 0,
      data[offset + 1] ?? 0,
      data[offset + 2] ?? 0,
      data[offset + 3] ?? 0,
    )
  }

  private static hasPcmExtensibleSubFormat(
    data: Uint8Array,
    fmtDataOffset: number,
    chunkSize: number,
  ): boolean {
    const pcmSubFormatGuid = [
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa,
      0x00, 0x38, 0x9b, 0x71,
    ]
    const subFormatOffset = fmtDataOffset + 24
    if (chunkSize < 40 || subFormatOffset + pcmSubFormatGuid.length > data.length) {
      return false
    }

    return pcmSubFormatGuid.every(
      (value, index) => data[subFormatOffset + index] === value,
    )
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
