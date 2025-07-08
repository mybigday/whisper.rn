import { SimulateFileAudioStreamAdapter } from '../SimulateFileAudioStreamAdapter'

// Mock WavFileReader
const mockWavFileReader = {
  initialize: jest.fn(),
  getHeader: jest.fn(),
  getAudioSlice: jest.fn(),
  getTotalDataSize: jest.fn(),
  byteToTime: jest.fn(),
  timeToByte: jest.fn(),
}

jest.mock('../../../utils/WavFileReader', () => ({
  WavFileReader: jest.fn().mockImplementation(() => mockWavFileReader),
}))

// Mock filesystem
const mockFs = {
  readFile: jest.fn(),
  exists: jest.fn(),
  unlink: jest.fn(),
}

describe('SimulateFileAudioStreamAdapter', () => {
  let adapter: SimulateFileAudioStreamAdapter
  let mockDataCallback: jest.Mock
  let mockErrorCallback: jest.Mock
  let mockStatusCallback: jest.Mock
  let mockOnEndOfFile: jest.Mock

  const sampleHeader = {
    sampleRate: 16000,
    channels: 1,
    bitsPerSample: 16,
    duration: 10.0,
  }

  beforeEach(() => {
    jest.clearAllMocks()
    jest.useFakeTimers()

    mockOnEndOfFile = jest.fn()
    mockDataCallback = jest.fn()
    mockErrorCallback = jest.fn()
    mockStatusCallback = jest.fn()

    // Setup default mocks
    mockWavFileReader.initialize.mockResolvedValue(undefined)
    mockWavFileReader.getHeader.mockReturnValue(sampleHeader)
    mockWavFileReader.getAudioSlice.mockReturnValue(new Uint8Array([1, 2, 3, 4]))
    mockWavFileReader.getTotalDataSize.mockReturnValue(16000)
    mockWavFileReader.byteToTime.mockImplementation((byte: number) => byte / 3200)
    mockWavFileReader.timeToByte.mockImplementation((time: number) => Math.floor(time * 3200))

    adapter = new SimulateFileAudioStreamAdapter({
      fs: mockFs,
      filePath: 'test.wav',
      onEndOfFile: mockOnEndOfFile,
    })
  })

  afterEach(async () => {
    jest.useRealTimers()
    await adapter.release()
  })

  describe('initialization', () => {
    it('should initialize with default options', async () => {
      const config = {
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      }

      await adapter.initialize(config)

      expect(mockWavFileReader.initialize).toHaveBeenCalled()
      expect(mockWavFileReader.getHeader).toHaveBeenCalled()
      expect(adapter.isRecording()).toBe(false)
    })

    it('should initialize with custom options', async () => {
      const customAdapter = new SimulateFileAudioStreamAdapter({
        fs: mockFs,
        filePath: 'custom.wav',
        playbackSpeed: 2.0,
        chunkDurationMs: 50,
        loop: true,
      })

      const config = {
        sampleRate: 44100,
        channels: 2,
        bitsPerSample: 16,
      }

      await customAdapter.initialize(config)

      expect(mockWavFileReader.initialize).toHaveBeenCalled()
      await customAdapter.release()
    })

    it('should handle initialization error', async () => {
      mockWavFileReader.initialize.mockRejectedValue(new Error('File not found'))

      adapter.onError(mockErrorCallback)

      await expect(adapter.initialize({})).rejects.toThrow('Failed to initialize SimulateFileAudioStreamAdapter: File not found')
      expect(mockErrorCallback).toHaveBeenCalledWith('File not found')
    })

    it('should handle missing header', async () => {
      mockWavFileReader.getHeader.mockReturnValue(null)

      adapter.onError(mockErrorCallback)

      await expect(adapter.initialize({})).rejects.toThrow('Failed to initialize SimulateFileAudioStreamAdapter: Failed to read WAV file header')
      expect(mockErrorCallback).toHaveBeenCalledWith('Failed to read WAV file header')
    })
  })

  describe('recording lifecycle', () => {
    beforeEach(async () => {
      await adapter.initialize({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      })
    })

    it('should start and stop recording', async () => {
      expect(adapter.isRecording()).toBe(false)

      await adapter.start()
      expect(adapter.isRecording()).toBe(true)

      await adapter.stop()
      expect(adapter.isRecording()).toBe(false)
    })

    it('should throw error when starting without initialization', async () => {
      const uninitializedAdapter = new SimulateFileAudioStreamAdapter({
        fs: mockFs,
        filePath: 'test.wav',
      })

      await expect(uninitializedAdapter.start()).rejects.toThrow('Adapter not initialized')
    })

    it('should handle multiple start calls gracefully', async () => {
      await adapter.start()
      expect(adapter.isRecording()).toBe(true)

      await adapter.start() // Should not throw
      expect(adapter.isRecording()).toBe(true)

      await adapter.stop()
    })

    it('should handle multiple stop calls gracefully', async () => {
      await adapter.start()
      await adapter.stop()
      expect(adapter.isRecording()).toBe(false)

      await adapter.stop() // Should not throw
      expect(adapter.isRecording()).toBe(false)
    })
  })

  describe('streaming behavior', () => {
    beforeEach(async () => {
      await adapter.initialize({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      })

      adapter.onData(mockDataCallback)
      adapter.onError(mockErrorCallback)
      adapter.onStatusChange(mockStatusCallback)
    })

    it('should call status callback on start/stop', async () => {
      await adapter.start()
      expect(mockStatusCallback).toHaveBeenCalledWith(true)

      await adapter.stop()
      expect(mockStatusCallback).toHaveBeenCalledWith(false)
    })

    it('should stream audio data at specified intervals', async () => {
      const chunkAdapter = new SimulateFileAudioStreamAdapter({
        fs: mockFs,
        filePath: 'test.wav',
        chunkDurationMs: 50,
      })

      await chunkAdapter.initialize({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      })

      chunkAdapter.onData(mockDataCallback)
      await chunkAdapter.start()

      // Fast forward time to trigger streaming
      jest.advanceTimersByTime(150)

      expect(mockDataCallback).toHaveBeenCalled()
      expect(mockWavFileReader.getAudioSlice).toHaveBeenCalled()

      await chunkAdapter.stop()
      await chunkAdapter.release()
    })

    it('should stop streaming when end of file is reached', async () => {
      mockWavFileReader.getAudioSlice.mockReturnValue(null)

      await adapter.start()

      jest.advanceTimersByTime(150)

      expect(mockOnEndOfFile).toHaveBeenCalled()
      expect(adapter.isRecording()).toBe(false)
    })

    it('should loop when configured', async () => {
      const loopAdapter = new SimulateFileAudioStreamAdapter({
        fs: mockFs,
        filePath: 'test.wav',
        loop: true,
        chunkDurationMs: 100,
      })

      await loopAdapter.initialize({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      })

      loopAdapter.onData(mockDataCallback)
      await loopAdapter.start()

      // Simulate end of file
      mockWavFileReader.getAudioSlice.mockReturnValueOnce(null)
      jest.advanceTimersByTime(150)

      // Should continue recording (loop)
      expect(loopAdapter.isRecording()).toBe(true)

      await loopAdapter.stop()
      await loopAdapter.release()
    })

    it('should handle streaming errors', async () => {
      mockWavFileReader.getAudioSlice.mockImplementation(() => {
        throw new Error('Read error')
      })

      await adapter.start()

      jest.advanceTimersByTime(150)

      expect(mockErrorCallback).toHaveBeenCalledWith('Read error')
      expect(adapter.isRecording()).toBe(false)
    })

    it('should create proper audio stream data', async () => {
      const testData = new Uint8Array([1, 2, 3, 4, 5, 6])
      mockWavFileReader.getAudioSlice.mockReturnValue(testData)

      await adapter.start()

      jest.advanceTimersByTime(150)

      expect(mockDataCallback).toHaveBeenCalledWith({
        data: testData,
        sampleRate: 16000,
        channels: 1,
        timestamp: expect.any(Number),
      })
    })
  })

  describe('playback control', () => {
    beforeEach(async () => {
      await adapter.initialize({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      })
    })

    it('should seek to time position', () => {
      adapter.seekToTime(5.0)
      expect(mockWavFileReader.timeToByte).toHaveBeenCalledWith(5.0)
    })

    it('should clamp seek time to valid range', () => {
      adapter.seekToTime(-1.0) // Should clamp to 0
      expect(mockWavFileReader.timeToByte).toHaveBeenCalledWith(0)

      adapter.seekToTime(15.0) // Should clamp to duration (10.0)
      expect(mockWavFileReader.timeToByte).toHaveBeenCalledWith(10.0)
    })

    it('should set playback speed', async () => {
      adapter.setPlaybackSpeed(2.0)

      // Should restart if currently playing
      await adapter.start()
      adapter.setPlaybackSpeed(0.5)

      // Wait for async restart to complete
      await Promise.resolve()
      expect(adapter.isRecording()).toBe(true)
    })

    it('should throw error for invalid playback speed', () => {
      expect(() => adapter.setPlaybackSpeed(0)).toThrow('Playback speed must be greater than 0')
      expect(() => adapter.setPlaybackSpeed(-1)).toThrow('Playback speed must be greater than 0')
    })

    it('should reset buffer to beginning', async () => {
      await adapter.start()

      adapter.resetBuffer()

      // Wait for async restart to complete
      await Promise.resolve()
      expect(adapter.isRecording()).toBe(true)
    })
  })

  describe('statistics', () => {
    beforeEach(async () => {
      await adapter.initialize({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      })
    })

    it('should return current statistics', () => {
      const stats = adapter.getStatistics()

      expect(stats).toEqual({
        filePath: 'test.wav',
        isRecording: false,
        currentTime: expect.any(Number),
        totalDuration: 10.0,
        progress: expect.any(Number),
        playbackSpeed: 1.0,
        currentBytePosition: 0,
        totalBytes: 16000,
        hasReachedEnd: false,
        header: sampleHeader,
      })
    })

    it('should update statistics during playback', async () => {
      await adapter.start()

      const stats = adapter.getStatistics()
      expect(stats.isRecording).toBe(true)

      await adapter.stop()
    })
  })

  describe('resource cleanup', () => {
    it('should release resources properly', async () => {
      await adapter.initialize({})
      await adapter.start()
      await adapter.release()

      expect(adapter.isRecording()).toBe(false)
    })

    it('should handle release when not initialized', async () => {
      const uninitializedAdapter = new SimulateFileAudioStreamAdapter({
        fs: mockFs,
        filePath: 'test.wav',
      })

      await expect(uninitializedAdapter.release()).resolves.not.toThrow()
    })

    it('should stop recording during release', async () => {
      await adapter.initialize({})
      await adapter.start()
      expect(adapter.isRecording()).toBe(true)

      await adapter.release()
      expect(adapter.isRecording()).toBe(false)
    })
  })

  describe('error handling', () => {
    beforeEach(async () => {
      await adapter.initialize({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      })
    })

    it('should handle missing header during streaming', async () => {
      mockWavFileReader.getHeader.mockReturnValue(null)

      adapter.onError(mockErrorCallback)
      await adapter.start()

      jest.advanceTimersByTime(150)

      expect(mockErrorCallback).toHaveBeenCalledWith('WAV file header not available')
    })

    it('should handle missing data callback', async () => {
      // Don't set data callback
      await adapter.start()

      jest.advanceTimersByTime(150)

      // Should not throw
      expect(adapter.isRecording()).toBe(true)
      await adapter.stop()
    })

    it('should handle missing config', async () => {
      // Don't set config
      await adapter.start()

      jest.advanceTimersByTime(150)

      // Should not throw
      expect(adapter.isRecording()).toBe(true)
      await adapter.stop()
    })
  })
})
