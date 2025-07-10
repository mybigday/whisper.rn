/* eslint-disable import/no-extraneous-dependencies */
// @ts-ignore
import LiveAudioStream from '@fugood/react-native-audio-pcm-stream'
import type { AudioStreamInterface, AudioStreamConfig, AudioStreamData } from '../types'
import { base64ToUint8Array } from '../../utils/common'

export class AudioPcmStreamAdapter implements AudioStreamInterface {
  private isInitialized = false

  private recording = false

  private config: AudioStreamConfig | null = null

  private dataCallback?: (data: AudioStreamData) => void

  private errorCallback?: (error: string) => void

  private statusCallback?: (isRecording: boolean) => void

  async initialize(config: AudioStreamConfig): Promise<void> {
    if (this.isInitialized) {
      await this.release()
    }

    try {
      this.config = config || null

      // Initialize LiveAudioStream
      LiveAudioStream.init({
        sampleRate: config.sampleRate || 16000,
        channels: config.channels || 1,
        bitsPerSample: config.bitsPerSample || 16,
        audioSource: config.audioSource || 6,
        bufferSize: config.bufferSize || 16 * 1024,
        wavFile: '', // We handle file writing separately
      })

      // Set up data listener
      LiveAudioStream.on('data', this.handleAudioData.bind(this))

      this.isInitialized = true
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown initialization error'
      this.errorCallback?.(errorMessage)
      throw new Error(`Failed to initialize LiveAudioStream: ${errorMessage}`)
    }
  }

  async start(): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('AudioStream not initialized')
    }

    if (this.recording) {
      return
    }

    try {
      LiveAudioStream.start()
      this.recording = true
      this.statusCallback?.(true)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown start error'
      this.errorCallback?.(errorMessage)
      throw new Error(`Failed to start recording: ${errorMessage}`)
    }
  }

  async stop(): Promise<void> {
    if (!this.recording) {
      return
    }

    try {
      await LiveAudioStream.stop()
      this.recording = false
      this.statusCallback?.(false)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown stop error'
      this.errorCallback?.(errorMessage)
      throw new Error(`Failed to stop recording: ${errorMessage}`)
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

    try {
      // LiveAudioStream doesn't have an explicit release method
      // But we should remove listeners and reset state
      this.isInitialized = false
      this.config = null
      this.dataCallback = undefined
      this.errorCallback = undefined
      this.statusCallback = undefined
    } catch (error) {
      console.warn('Error during LiveAudioStream release:', error)
    }
  }

  /**
   * Handle incoming audio data from LiveAudioStream
   */
  private handleAudioData(base64Data: string): void {
    if (!this.dataCallback) {
      return
    }

    try {
      const audioData = base64ToUint8Array(base64Data)

      const streamData: AudioStreamData = {
        data: audioData,
        sampleRate: this.config?.sampleRate || 16000,
        channels: this.config?.channels || 1,
        timestamp: Date.now(),
      }

      this.dataCallback(streamData)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Audio processing error'
      this.errorCallback?.(errorMessage)
    }
  }
}
