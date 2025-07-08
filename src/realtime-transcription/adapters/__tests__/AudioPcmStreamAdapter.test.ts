import LiveAudioStream from '@fugood/react-native-audio-pcm-stream'
import { AudioPcmStreamAdapter } from '../AudioPcmStreamAdapter'

// Create persistent mock callbacks map
const mockCallbacks = new Map()

// Mock the LiveAudioStream library
jest.mock('@fugood/react-native-audio-pcm-stream', () => ({
  init: jest.fn(),
  start: jest.fn(),
  stop: jest.fn().mockResolvedValue(undefined),
  on: jest.fn((event: string, callback: any) => {
    mockCallbacks.set(event, callback)
  }),
  off: jest.fn((event: string) => {
    mockCallbacks.delete(event)
  }),
  getCallback: (event: string) => mockCallbacks.get(event), // Helper for tests
}))

// Mock the base64ToUint8Array utility
jest.mock('../../../utils/common', () => ({
  base64ToUint8Array: jest.fn(
    (base64: string) => new Uint8Array(Buffer.from(base64, 'base64')),
  ),
}))

// Import the mocked module
const mockLiveAudioStream = LiveAudioStream as jest.Mocked<
  typeof LiveAudioStream
>

describe('AudioPcmStreamAdapter', () => {
  let adapter: AudioPcmStreamAdapter
  let mockDataCallback: jest.Mock
  let mockErrorCallback: jest.Mock
  let mockStatusCallback: jest.Mock

  beforeEach(() => {
    // @ts-ignore
    mockLiveAudioStream.off('data')
    jest.clearAllMocks()
    // Clear the callbacks map
    mockCallbacks.clear()

    // Reset mock implementations to successful defaults
    mockLiveAudioStream.init.mockImplementation(() => {})
    mockLiveAudioStream.start.mockImplementation(() => {})
    mockLiveAudioStream.stop.mockResolvedValue(undefined as any)

    adapter = new AudioPcmStreamAdapter()
    mockDataCallback = jest.fn()
    mockErrorCallback = jest.fn()
    mockStatusCallback = jest.fn()
  })

  afterEach(async () => {
    await adapter.release()
  })

  describe('initialization', () => {
    it('should initialize with default config', async () => {
      const config = {
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      }

      await adapter.initialize(config)

      expect(mockLiveAudioStream.init).toHaveBeenCalledWith({
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
        audioSource: 6,
        bufferSize: 16 * 1024,
        wavFile: '',
      })
      expect(mockLiveAudioStream.on).toHaveBeenCalledWith(
        'data',
        expect.any(Function),
      )
    })

    it('should initialize with custom config', async () => {
      const config = {
        sampleRate: 44100,
        channels: 2,
        bitsPerSample: 32,
        audioSource: 1,
        bufferSize: 8 * 1024,
      }

      await adapter.initialize(config)

      expect(mockLiveAudioStream.init).toHaveBeenCalledWith({
        sampleRate: 44100,
        channels: 2,
        bitsPerSample: 32,
        audioSource: 1,
        bufferSize: 8 * 1024,
        wavFile: '',
      })
    })

    it('should handle initialization error', async () => {
      mockLiveAudioStream.init.mockImplementation(() => {
        throw new Error('Initialization failed')
      })

      adapter.onError(mockErrorCallback)

      await expect(adapter.initialize({})).rejects.toThrow(
        'Failed to initialize LiveAudioStream: Initialization failed',
      )
      expect(mockErrorCallback).toHaveBeenCalledWith('Initialization failed')
    })

    it('should release existing instance before re-initialization', async () => {
      await adapter.initialize({})
      await adapter.start()

      // Re-initialize should release the previous instance
      await adapter.initialize({})

      expect(mockLiveAudioStream.stop).toHaveBeenCalled()
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

    it('should start recording', async () => {
      expect(adapter.isRecording()).toBe(false)

      await adapter.start()

      expect(mockLiveAudioStream.start).toHaveBeenCalled()
      expect(adapter.isRecording()).toBe(true)
    })

    it('should stop recording', async () => {
      await adapter.start()
      expect(adapter.isRecording()).toBe(true)

      await adapter.stop()

      expect(mockLiveAudioStream.stop).toHaveBeenCalled()
      expect(adapter.isRecording()).toBe(false)
    })

    it('should throw error when starting without initialization', async () => {
      const uninitializedAdapter = new AudioPcmStreamAdapter()
      await expect(uninitializedAdapter.start()).rejects.toThrow(
        'AudioStream not initialized',
      )
    })

    it('should handle multiple start calls gracefully', async () => {
      await adapter.start()
      await adapter.start() // Should not throw or call start again

      expect(mockLiveAudioStream.start).toHaveBeenCalledTimes(1)
    })

    it('should handle multiple stop calls gracefully', async () => {
      await adapter.start()
      await adapter.stop()
      await adapter.stop() // Should not throw or call stop again

      expect(mockLiveAudioStream.stop).toHaveBeenCalledTimes(1)
    })

    it('should handle start error', async () => {
      mockLiveAudioStream.start.mockImplementation(() => {
        throw new Error('Start failed')
      })

      adapter.onError(mockErrorCallback)

      await expect(adapter.start()).rejects.toThrow(
        'Failed to start recording: Start failed',
      )
      expect(mockErrorCallback).toHaveBeenCalledWith('Start failed')
    })

    it('should handle stop error', async () => {
      await adapter.start()

      mockLiveAudioStream.stop.mockImplementation(() => {
        throw new Error('Stop failed')
      })

      adapter.onError(mockErrorCallback)

      await expect(adapter.stop()).rejects.toThrow(
        'Failed to stop recording: Stop failed',
      )
      expect(mockErrorCallback).toHaveBeenCalledWith('Stop failed')

      // Reset the mock for cleanup
      mockLiveAudioStream.stop.mockResolvedValue(undefined as any)
    })
  })

  describe('callback handling', () => {
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

    it('should process audio data from LiveAudioStream', async () => {
      const base64Data = 'SGVsbG8gV29ybGQ=' // "Hello World" in base64

      // Get the data handler that was registered with LiveAudioStream
      const dataHandler = mockLiveAudioStream.on.mock.calls.find(
        (call) => call[0] === 'data',
      )?.[1]
      expect(dataHandler).toBeDefined()

      // Simulate receiving data from LiveAudioStream
      dataHandler!(base64Data)

      expect(mockDataCallback).toHaveBeenCalledWith({
        data: expect.any(Uint8Array),
        sampleRate: 16000,
        channels: 1,
        timestamp: expect.any(Number),
      })
    })

    it('should handle audio data processing error', async () => {
      const { base64ToUint8Array } = require('../../../utils/common')
      base64ToUint8Array.mockImplementation(() => {
        throw new Error('Base64 decode failed')
      })

      const dataHandler = mockLiveAudioStream.on.mock.calls.find(
        (call) => call[0] === 'data',
      )?.[1]

      dataHandler!('invalid_base64')

      expect(mockErrorCallback).toHaveBeenCalledWith('Base64 decode failed')
    })

    it('should ignore data when no callback is set', async () => {
      const adapterWithoutCallback = new AudioPcmStreamAdapter()
      await adapterWithoutCallback.initialize({})

      const dataHandler = mockLiveAudioStream.on.mock.calls.find(
        (call) => call[0] === 'data',
      )?.[1]

      // Should not throw
      expect(() => dataHandler!('test_data')).not.toThrow()
      await adapterWithoutCallback.release()
    })
  })

  describe('resource cleanup', () => {
    it('should release resources properly', async () => {
      await adapter.initialize({})
      await adapter.start()
      await adapter.release()

      expect(adapter.isRecording()).toBe(false)
      expect(mockLiveAudioStream.stop).toHaveBeenCalled()
    })

    it('should handle release when not initialized', async () => {
      const uninitializedAdapter = new AudioPcmStreamAdapter()
      await expect(uninitializedAdapter.release()).resolves.not.toThrow()
    })

    it('should stop recording during release', async () => {
      await adapter.initialize({})
      await adapter.start()
      expect(adapter.isRecording()).toBe(true)

      await adapter.release()
      expect(adapter.isRecording()).toBe(false)
    })

    it('should clear all callbacks on release', async () => {
      await adapter.initialize({})
      adapter.onData(mockDataCallback)
      adapter.onError(mockErrorCallback)
      adapter.onStatusChange(mockStatusCallback)

      await adapter.release()

      // After release, callbacks should be cleared
      const dataHandler = mockLiveAudioStream.on.mock.calls.find(
        (call) => call[0] === 'data',
      )?.[1]
      expect(() => dataHandler!('test_data')).not.toThrow()
    })
  })

})
