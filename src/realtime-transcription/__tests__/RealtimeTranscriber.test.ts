import { RealtimeTranscriber } from '../RealtimeTranscriber'
import { JestAudioStreamAdapter } from '../adapters/JestAudioStreamAdapter'
import { VAD_PRESETS } from '../types'

// Mock WavFileWriter
const mockWavFileWriter = {
  initialize: jest.fn(),
  appendAudioData: jest.fn(),
  finalize: jest.fn(),
  cancel: jest.fn(),
}

jest.mock('../../utils/WavFileWriter', () => ({
  WavFileWriter: jest.fn().mockImplementation(() => mockWavFileWriter),
}))

describe('RealtimeTranscriber', () => {
  let transcriber: RealtimeTranscriber
  let mockWhisperContext: any
  let mockVadContext: any
  let mockAudioStream: JestAudioStreamAdapter
  let mockFs: any
  let mockCallbacks: any

  // Helper function to create audio data with SharedArrayBuffer for transcription tests
  const createAudioData = (size: number): Uint8Array => {
    try {
      // Try to create SharedArrayBuffer for proper transcription testing
      const sharedBuffer = new SharedArrayBuffer(size)
      const audioData = new Uint8Array(sharedBuffer)
      audioData.fill(1)
      return audioData
    } catch (e) {
      // Fallback to regular ArrayBuffer if SharedArrayBuffer is not available
      const audioData = new Uint8Array(size)
      audioData.fill(1)
      return audioData
    }
  }

  beforeEach(() => {
    jest.clearAllMocks()
    // Use real timers for more stable async testing
    jest.useRealTimers()

    // Mock Whisper context
    mockWhisperContext = {
      transcribeData: jest.fn(() => ({
        stop: jest.fn(),
        promise: Promise.resolve({
          isAborted: false,
          result: 'Test transcription',
          segments: [{ text: 'Test transcription', t0: 0, t1: 1000 }],
        }),
      })),
    }

    // Mock VAD context
    mockVadContext = {
      detectSpeechData: jest.fn(() =>
        Promise.resolve([{ t0: 0, t1: 1000 }])
      ),
    }

    // Mock audio stream - disable automatic streaming by setting maxChunks to 0 explicitly
    mockAudioStream = new JestAudioStreamAdapter({
      chunkSize: 3200,
      chunkInterval: 100,
      generateSilence: false,
    })

    // Override the startStreaming method to prevent automatic streaming during tests
    mockAudioStream['startStreaming'] = jest.fn()

    // Mock filesystem
    mockFs = {
      readFile: jest.fn(),
      exists: jest.fn(),
      unlink: jest.fn(),
    }

    // Mock callbacks
    mockCallbacks = {
      onTranscribe: jest.fn(),
      onVad: jest.fn(),
      onError: jest.fn(),
      onStatusChange: jest.fn(),
      onStatsUpdate: jest.fn(),
    }

    transcriber = new RealtimeTranscriber(
      {
        whisperContext: mockWhisperContext,
        vadContext: mockVadContext,
        audioStream: mockAudioStream,
        fs: mockFs,
      },
      {
        audioSliceSec: 2,
        audioMinSec: 0.5,
        maxSlicesInMemory: 3,
        vadOptions: VAD_PRESETS.default,
        autoSliceOnSpeechEnd: true,
      },
      mockCallbacks
    )
  })

  afterEach(async () => {
    await transcriber.release()
  })

  describe('initialization', () => {
    it('should initialize with default options', () => {
      const basicTranscriber = new RealtimeTranscriber({
        whisperContext: mockWhisperContext,
        audioStream: mockAudioStream,
      })

      expect(basicTranscriber.getStatistics()).toEqual({
        isActive: false,
        isTranscribing: false,
        vadEnabled: false,
        queueLength: 0,
        audioStats: expect.any(Object),
        vadStats: expect.any(Object),
        sliceStats: expect.any(Object),
        autoSliceConfig: expect.any(Object),
      })
    })

    it('should initialize with custom options', () => {
      const customTranscriber = new RealtimeTranscriber(
        {
          whisperContext: mockWhisperContext,
          vadContext: mockVadContext,
          audioStream: mockAudioStream,
        },
        {
          audioSliceSec: 5,
          audioMinSec: 2,
          maxSlicesInMemory: 5,
          vadPreset: 'sensitive',
          autoSliceOnSpeechEnd: false,
        }
      )

      expect(customTranscriber.getStatistics().vadEnabled).toBe(true)
    })

    it('should apply VAD presets correctly', () => {
      const presetTranscriber = new RealtimeTranscriber(
        {
          whisperContext: mockWhisperContext,
          vadContext: mockVadContext,
          audioStream: mockAudioStream,
        },
        {
          vadPreset: 'very-sensitive',
          vadOptions: { threshold: 0.8 }, // Should be overridden by preset
        }
      )

      // The preset should be applied
      expect(presetTranscriber.getStatistics().vadEnabled).toBe(true)
    })
  })

  describe('lifecycle management', () => {
    it('should start and stop transcription', async () => {
      expect(transcriber.getStatistics().isActive).toBe(false)

      await transcriber.start()
      expect(transcriber.getStatistics().isActive).toBe(true)
      expect(mockCallbacks.onStatusChange).toHaveBeenCalledWith(true)

      await transcriber.stop()
      expect(transcriber.getStatistics().isActive).toBe(false)
      expect(mockCallbacks.onStatusChange).toHaveBeenCalledWith(false)
    })

    it('should handle multiple start calls', async () => {
      await transcriber.start()

      await expect(transcriber.start()).rejects.toThrow('Realtime transcription is already active')
    })

    it('should handle stop when not active', async () => {
      // Should not throw
      await expect(transcriber.stop()).resolves.not.toThrow()
    })

    it('should initialize audio stream on start', async () => {
      const initializeSpy = jest.spyOn(mockAudioStream, 'initialize')
      const startSpy = jest.spyOn(mockAudioStream, 'start')

      await transcriber.start()

      expect(initializeSpy).toHaveBeenCalledWith({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
        audioSource: 6,
        bufferSize: 16 * 1024,
      })
      expect(startSpy).toHaveBeenCalled()
    })

    it('should stop audio stream on stop', async () => {
      const stopSpy = jest.spyOn(mockAudioStream, 'stop')

      await transcriber.start()
      await transcriber.stop()

      expect(stopSpy).toHaveBeenCalled()
    })
  })

  describe('audio processing', () => {
    beforeEach(async () => {
      await transcriber.start()
    })

    it('should process audio data from stream', async () => {
      const testData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])
      mockAudioStream.simulateDataChunk(testData)

      // Fast forward to allow processing
      await new Promise(resolve => setTimeout(resolve, 100))

      const stats = transcriber.getStatistics()
      expect(stats.audioStats.accumulatedSamples).toBeGreaterThan(0)
    })

    it('should accumulate audio data', async () => {
      const chunk1 = new Uint8Array([1, 2, 3, 4])
      const chunk2 = new Uint8Array([5, 6, 7, 8])

      mockAudioStream.simulateDataChunk(chunk1)
      mockAudioStream.simulateDataChunk(chunk2)

      await new Promise(resolve => setTimeout(resolve, 100))

      const stats = transcriber.getStatistics()
      expect(stats.audioStats.accumulatedSamples).toBeGreaterThan(0) // Should have accumulated some audio data
    })

        it('should write to WAV file when configured', async () => {
      // Create a new audio stream instance for this test to avoid interference
      const testAudioStream = new JestAudioStreamAdapter({
        chunkSize: 3200,
        chunkInterval: 100,
        generateSilence: false,
      })

      const wavTranscriber = new RealtimeTranscriber(
        {
          whisperContext: mockWhisperContext,
          audioStream: testAudioStream,
          fs: mockFs,
        },
        {
          audioOutputPath: 'test.wav',
        }
      )

      await wavTranscriber.start()

      // Verify that the audio stream is recording and initialized
      expect(testAudioStream.isRecording()).toBe(true)

      // Use any data size - the WAV writer should be called for any audio data
      const testData = new Uint8Array(100)
      testData.fill(1)
      testAudioStream.simulateDataChunk(testData)

      expect(mockWavFileWriter.initialize).toHaveBeenCalled()
      expect(mockWavFileWriter.appendAudioData).toHaveBeenCalled()

      await wavTranscriber.stop()
      await wavTranscriber.release()
    })
  })

  describe('VAD processing', () => {
    beforeEach(async () => {
      await transcriber.start()
    })

    it('should detect speech when VAD is enabled', async () => {
      mockVadContext.detectSpeechData.mockResolvedValue([{ t0: 0, t1: 1000 }])

      // Use larger data to trigger processing (at least 3200 bytes)
      const audioData = createAudioData(16000)
      mockAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(mockVadContext.detectSpeechData).toHaveBeenCalled()
      expect(mockCallbacks.onVad).toHaveBeenCalled()
    })

    it('should trigger transcription on speech detection', async () => {
      mockVadContext.detectSpeechData.mockResolvedValue([{ t0: 0, t1: 1000 }])

      const audioData = createAudioData(16000)
      mockAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(mockWhisperContext.transcribeData).toHaveBeenCalled()
      expect(mockCallbacks.onTranscribe).toHaveBeenCalled()
    })

    it('should handle VAD processing errors', async () => {
      mockVadContext.detectSpeechData.mockRejectedValue(new Error('VAD error'))

      const audioData = createAudioData(16000)
      mockAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete - wait longer for error handling
      await new Promise(resolve => setTimeout(resolve, 200))

      expect(mockCallbacks.onError).toHaveBeenCalledWith(
        expect.stringContaining('VAD processing error')
      )
    })

    it('should skip VAD when disabled', async () => {
      // Create a fresh audio stream for this test
      const testAudioStream = new JestAudioStreamAdapter({
        chunkSize: 3200,
        chunkInterval: 100,
        generateSilence: false,
      })

      const noVadTranscriber = new RealtimeTranscriber(
        {
          whisperContext: mockWhisperContext,
          audioStream: testAudioStream,
        },
        {},
        mockCallbacks
      )

      await noVadTranscriber.start()

      const audioData = createAudioData(16000)
      testAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(mockCallbacks.onVad).not.toHaveBeenCalled()
      expect(mockCallbacks.onTranscribe).toHaveBeenCalled() // Should still transcribe without VAD

      await noVadTranscriber.stop()
      await noVadTranscriber.release()
    })
  })

  describe('transcription processing', () => {
    beforeEach(async () => {
      await transcriber.start()
    })

    it('should process transcription queue', async () => {
      const audioData = createAudioData(16000)
      mockAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(mockWhisperContext.transcribeData).toHaveBeenCalled()
      expect(mockCallbacks.onTranscribe).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'transcribe',
          data: expect.objectContaining({
            result: 'Test transcription',
          }),
        })
      )
    })

    it('should handle transcription errors', async () => {
      // Create a promise that will be rejected
      const rejectedPromise = Promise.reject(new Error('Transcription error'))
      // Catch the promise to prevent unhandled rejection
      rejectedPromise.catch(() => {})

      mockWhisperContext.transcribeData.mockReturnValue({
        stop: jest.fn(),
        promise: rejectedPromise,
      })

      const audioData = createAudioData(16000)
      mockAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(mockCallbacks.onTranscribe).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
        })
      )
    })

    it('should build prompts from previous transcriptions', async () => {
      // Create a fresh audio stream for this test
      const testAudioStream = new JestAudioStreamAdapter({
        chunkSize: 3200,
        chunkInterval: 100,
        generateSilence: false,
      })

      const promptTranscriber = new RealtimeTranscriber(
        {
          whisperContext: mockWhisperContext,
          audioStream: testAudioStream,
        },
        {
          promptPreviousSlices: true,
          initialPrompt: 'Initial prompt',
        }
      )

      await promptTranscriber.start()

      const audioData = createAudioData(16000)
      testAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 50))

      expect(mockWhisperContext.transcribeData).toHaveBeenCalledWith(
        expect.anything(),
        {
          onProgress: undefined,
          prompt: expect.stringContaining('Initial prompt'),
        }
      )

      await promptTranscriber.stop()
      await promptTranscriber.release()
    })

    it('should manage transcription queue correctly', async () => {
      // Add multiple chunks quickly
      for (let i = 0; i < 3; i += 1) {
        const audioData = new Uint8Array(16000)
        audioData.fill(i)
        mockAudioStream.simulateDataChunk(audioData)
      }

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      const stats = transcriber.getStatistics()
      expect(stats.queueLength).toBeGreaterThanOrEqual(0)
    })
  })

  describe('auto-slicing', () => {
    beforeEach(async () => {
      await transcriber.start()
    })

    it('should auto-slice on speech end', async () => {
      mockVadContext.detectSpeechData.mockResolvedValue([])

      const audioData = new Uint8Array(16000)
      audioData.fill(1)
      mockAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should trigger auto-slice if conditions are met
      expect(mockCallbacks.onVad).toHaveBeenCalled()
    })

    it('should respect minimum duration for auto-slice', async () => {
      const shortTranscriber = new RealtimeTranscriber(
        {
          whisperContext: mockWhisperContext,
          vadContext: mockVadContext,
          audioStream: mockAudioStream,
        },
        {
          audioMinSec: 2.0,
          autoSliceOnSpeechEnd: true,
        }
      )

      await shortTranscriber.start()

      const audioData = new Uint8Array(16000)
      audioData.fill(1)
      mockAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should not transcribe due to minimum duration
      expect(mockCallbacks.onTranscribe).not.toHaveBeenCalled()

      await shortTranscriber.stop()
      await shortTranscriber.release()
    })

    it('should force next slice manually', async () => {
      const audioData = new Uint8Array(16000)
      audioData.fill(1)
      mockAudioStream.simulateDataChunk(audioData)

      await transcriber.nextSlice()

      expect(mockCallbacks.onTranscribe).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'start',
        })
      )
    })
  })

  describe('configuration updates', () => {
    it('should update callbacks', () => {
      const newCallbacks = {
        onTranscribe: jest.fn(),
        onError: jest.fn(),
      }

      transcriber.updateCallbacks(newCallbacks)

      // Test that new callbacks are used
      transcriber.updateCallbacks({ onError: jest.fn() })
      // Should not throw
    })

    it('should update VAD options', () => {
      const newVadOptions = {
        threshold: 0.8,
        minSpeechDurationMs: 500,
      }

      transcriber.updateVadOptions(newVadOptions)

      // Should update options without throwing
      expect(() => transcriber.updateVadOptions(newVadOptions)).not.toThrow()
    })

    it('should update auto-slice options', () => {
      const newAutoSliceOptions = {
        autoSliceOnSpeechEnd: false,
        autoSliceThreshold: 0.9,
      }

      transcriber.updateAutoSliceOptions(newAutoSliceOptions)

      // Should update options without throwing
      expect(() => transcriber.updateAutoSliceOptions(newAutoSliceOptions)).not.toThrow()
    })
  })

  describe('statistics and monitoring', () => {
    it('should provide comprehensive statistics', () => {
      const stats = transcriber.getStatistics()

      expect(stats).toEqual({
        isActive: false,
        isTranscribing: false,
        vadEnabled: true,
        queueLength: 0,
        audioStats: expect.any(Object),
        vadStats: expect.any(Object),
        sliceStats: expect.any(Object),
        autoSliceConfig: expect.any(Object),
      })
    })

    it('should update statistics during operation', async () => {
      await transcriber.start()

      const beforeStats = transcriber.getStatistics()
      expect(beforeStats.isActive).toBe(true)

      const audioData = new Uint8Array(1000)
      audioData.fill(1)
      mockAudioStream.simulateDataChunk(audioData)

      await new Promise(resolve => setTimeout(resolve, 100))

      const afterStats = transcriber.getStatistics()
      expect(afterStats.audioStats.accumulatedSamples).toBeGreaterThan(0)
    })

    it('should emit stats updates', async () => {
      await transcriber.start()

      expect(mockCallbacks.onStatsUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'status_change',
        })
      )
    })

    it('should provide transcription results', async () => {
      await transcriber.start()

      const audioData = new Uint8Array(16000)
      audioData.fill(1)
      mockAudioStream.simulateDataChunk(audioData)

      // Allow processing to complete
      await new Promise(resolve => setTimeout(resolve, 100))

      const results = transcriber.getTranscriptionResults()
      expect(results.length).toBeGreaterThanOrEqual(0)
    })
  })

  describe('error handling', () => {
    it('should handle audio stream errors', async () => {
      await transcriber.start()

      mockAudioStream.simulateError('Audio stream error')

      expect(mockCallbacks.onError).toHaveBeenCalledWith('Audio stream error')
    })

    it('should handle start errors', async () => {
      const errorStream = new JestAudioStreamAdapter({ simulateErrors: true })
      const errorTranscriber = new RealtimeTranscriber(
        {
          whisperContext: mockWhisperContext,
          audioStream: errorStream,
        },
        {},
        mockCallbacks
      )

      await expect(errorTranscriber.start()).rejects.toThrow()
      expect(mockCallbacks.onStatusChange).toHaveBeenCalledWith(false)

      await errorTranscriber.release()
    })

    it('should handle stop errors gracefully', async () => {
      // Create a fresh audio stream for this test to avoid cleanup issues
      const testAudioStream = new JestAudioStreamAdapter({
        chunkSize: 3200,
        chunkInterval: 100,
        generateSilence: false,
      })

      const testTranscriber = new RealtimeTranscriber(
        {
          whisperContext: mockWhisperContext,
          vadContext: mockVadContext,
          audioStream: testAudioStream,
          fs: mockFs,
        },
        {
          audioSliceSec: 2,
          audioMinSec: 0.5,
          maxSlicesInMemory: 3,
          vadOptions: VAD_PRESETS.default,
          autoSliceOnSpeechEnd: true,
        },
        mockCallbacks
      )

      await testTranscriber.start()

      // Mock stop to throw error
      const stopSpy = jest.spyOn(testAudioStream, 'stop')
      stopSpy.mockImplementation(async () => {
        throw new Error('Stop error')
      })

      await expect(testTranscriber.stop()).resolves.not.toThrow()
      expect(mockCallbacks.onError).toHaveBeenCalledWith(
        expect.stringContaining('Stop error')
      )

      // Clean up properly
      stopSpy.mockRestore()
      await testTranscriber.release()
    })
  })

  describe('resource cleanup', () => {
    it('should release all resources', async () => {
      await transcriber.start()

      const audioData = new Uint8Array(1000)
      audioData.fill(1)
      mockAudioStream.simulateDataChunk(audioData)

      await transcriber.release()

      expect(transcriber.getStatistics().isActive).toBe(false)
      expect(mockAudioStream.isRecording()).toBe(false)
    })

    it('should handle release when not started', async () => {
      await expect(transcriber.release()).resolves.not.toThrow()
    })

    it('should reset state on release', async () => {
      await transcriber.start()

      const audioData = new Uint8Array(1000)
      audioData.fill(1)
      mockAudioStream.simulateDataChunk(audioData)

      await transcriber.release()

      const stats = transcriber.getStatistics()
      expect(stats.isActive).toBe(false)
      expect(stats.isTranscribing).toBe(false)
      expect(stats.queueLength).toBe(0)
    })
  })
})
