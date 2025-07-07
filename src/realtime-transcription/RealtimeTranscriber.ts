/* eslint-disable class-methods-use-this */
import type { WhisperContext, WhisperVadContext, VadOptions } from '../index'
import { SliceManager } from './SliceManager'
import { WavFileWriter, WavFileWriterFs } from '../utils/WavFileWriter'
import type {
  RealtimeOptions,
  TranscribeEvent,
  VADEvent,
  RealtimeTranscriberCallbacks,
  StatsEvent,
  RealtimeTranscriberDependencies,
  AudioStreamData,
  AudioSliceNoData,
  AudioStreamInterface,
} from './types'
import { VAD_PRESETS } from './types'

/**
 * RealtimeTranscriber provides real-time audio transcription with VAD support.
 *
 * Features:
 * - Automatic slice management based on duration
 * - VAD-based speech detection and auto-slicing
 * - Configurable auto-slice mechanism that triggers on speech_end/silence events
 * - Memory management for audio slices
 * - Queue-based transcription processing
 */
export class RealtimeTranscriber {
  private whisperContext: WhisperContext

  private vadContext?: WhisperVadContext

  private audioStream: AudioStreamInterface

  private sliceManager: SliceManager

  private callbacks: RealtimeTranscriberCallbacks = {}

  private options: {
    audioSliceSec: number
    audioMinSec: number
    maxSlicesInMemory: number
    vadOptions: VadOptions
    vadPreset?: keyof typeof VAD_PRESETS
    autoSliceOnSpeechEnd: boolean
    autoSliceThreshold: number
    transcribeOptions: any
    initialPrompt?: string
    promptPreviousSlices: boolean
    fs?: WavFileWriterFs
    audioOutputPath?: string
  }

  private isActive = false

  private isTranscribing = false

  private vadEnabled = false

  private transcriptionQueue: Array<{
    sliceIndex: number
    audioData: Uint8Array
  }> = []

  private accumulatedData: Uint8Array = new Uint8Array(0)

  private wavFileWriter: WavFileWriter | null = null

  // Simplified VAD state management
  private lastSpeechDetectedTime = 0

  // Track VAD state for proper event transitions
  private lastVadState: 'speech' | 'silence' = 'silence'

  // Track last stats to emit only when changed
  private lastStatsSnapshot: any = null

  // Store transcription results by slice index
  private transcriptionResults: Map<
    number,
    { slice: AudioSliceNoData; transcribeEvent: TranscribeEvent }
  > = new Map()

  constructor(
    dependencies: RealtimeTranscriberDependencies,
    options: RealtimeOptions = {},
    callbacks: RealtimeTranscriberCallbacks = {},
  ) {
    this.whisperContext = dependencies.contexts.whisperContext
    this.vadContext = dependencies.contexts.vadContext
    this.audioStream = dependencies.audioStream
    this.callbacks = callbacks

    // Set default options with proper types
    this.options = {
      audioSliceSec: options.audioSliceSec || 30,
      audioMinSec: options.audioMinSec || 1,
      maxSlicesInMemory: options.maxSlicesInMemory || 3,
      vadOptions: options.vadOptions || VAD_PRESETS.DEFAULT,
      vadPreset: options.vadPreset,
      autoSliceOnSpeechEnd: options.autoSliceOnSpeechEnd || true,
      autoSliceThreshold: options.autoSliceThreshold || 0.5,
      transcribeOptions: options.transcribeOptions || {},
      initialPrompt: options.initialPrompt,
      promptPreviousSlices: options.promptPreviousSlices ?? true,
      audioOutputPath: options.audioOutputPath,
    }

    // Apply VAD preset if specified
    if (this.options.vadPreset && VAD_PRESETS[this.options.vadPreset]) {
      this.options.vadOptions = {
        ...VAD_PRESETS[this.options.vadPreset],
        ...this.options.vadOptions,
      }
    }

    // Enable VAD if context is provided and not explicitly disabled
    this.vadEnabled = !!this.vadContext

    // Initialize managers
    this.sliceManager = new SliceManager(
      this.options.audioSliceSec,
      this.options.maxSlicesInMemory,
    )

    // Set up audio stream callbacks
    this.audioStream.onData(this.handleAudioData.bind(this))
    this.audioStream.onError(this.handleError.bind(this))
    this.audioStream.onStatusChange(this.handleAudioStatusChange.bind(this))
  }

  /**
   * Start realtime transcription
   */
  async start(): Promise<void> {
    if (this.isActive) {
      throw new Error('Realtime transcription is already active')
    }

    try {
      this.isActive = true
      this.callbacks.onStatusChange?.(true)

      // Reset all state to ensure clean start
      this.reset()

      // Initialize WAV file writer if output path is specified
      if (this.options.fs && this.options.audioOutputPath) {
        this.wavFileWriter = new WavFileWriter(
          this.options.fs,
          this.options.audioOutputPath,
          {
            sampleRate: 16000, // Default sample rate
            channels: 1,
            bitsPerSample: 16,
          },
        )
        await this.wavFileWriter.initialize()
      }

      // Start audio recording
      await this.audioStream.start()

      // Emit stats update for status change
      this.emitStatsUpdate('status_change')

      this.log('Realtime transcription started')
    } catch (error) {
      this.isActive = false
      this.callbacks.onStatusChange?.(false)
      throw error
    }
  }

  /**
   * Stop realtime transcription
   */
  async stop(): Promise<void> {
    if (!this.isActive) {
      return
    }

    try {
      this.isActive = false

      // Stop audio recording
      await this.audioStream.stop()

      // Process any remaining accumulated data
      if (this.accumulatedData.length > 0) {
        this.processAccumulatedDataForSliceManagement()
      }

      // Process any remaining queued transcriptions
      await this.processTranscriptionQueue()

      // Finalize WAV file
      if (this.wavFileWriter) {
        await this.wavFileWriter.finalize()
        this.wavFileWriter = null
      }

      // Reset all state completely
      this.reset()

      this.callbacks.onStatusChange?.(false)

      // Emit stats update for status change
      this.emitStatsUpdate('status_change')

      this.log('Realtime transcription stopped')
    } catch (error) {
      this.handleError(`Stop error: ${error}`)
    }
  }

  /**
   * Handle incoming audio data from audio stream
   */
  private handleAudioData(streamData: AudioStreamData): void {
    if (!this.isActive) {
      return
    }

    try {
      // Write to WAV file if enabled (convert to Uint8Array for WavFileWriter)
      if (this.wavFileWriter) {
        this.wavFileWriter.appendAudioData(streamData.data).catch((error) => {
          console.warn('Failed to write audio to WAV file:', error)
        })
      }

      // Always accumulate data for slice management
      this.accumulateAudioData(streamData.data)
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Audio processing error'
      this.handleError(errorMessage)
    }
  }

  /**
   * Accumulate audio data for slice management
   */
  private accumulateAudioData(newData: Uint8Array): void {
    const combined = new Uint8Array(
      this.accumulatedData.length + newData.length,
    )
    combined.set(this.accumulatedData)
    combined.set(new Uint8Array(newData), this.accumulatedData.length)
    this.accumulatedData = combined

    // Process accumulated data when we have enough for slice management
    const minBufferSamples = 16000 * 1 // 1 second for slice management
    if (this.accumulatedData.length >= minBufferSamples) {
      this.processAccumulatedDataForSliceManagement()
    }
  }

  /**
   * Process accumulated audio data through SliceManager
   */
  private processAccumulatedDataForSliceManagement(): void {
    if (this.accumulatedData.length === 0) {
      return
    }

    // Process through slice manager directly with Uint8Array
    const result = this.sliceManager.addAudioData(this.accumulatedData)

    if (result.slice) {
      this.log(
        `Slice ${result.slice.index} ready (${result.slice.data.length} bytes)`,
      )

      // Process VAD for the slice if enabled
      if (!this.isTranscribing && this.vadEnabled) {
        this.processSliceVAD(result.slice).catch((error: any) => {
          this.handleError(`VAD processing error: ${error}`)
        })
      } else if (!this.isTranscribing) {
        // If VAD is disabled, transcribe slices as they become ready
        this.queueSliceForTranscription(result.slice).catch((error: any) => {
          this.handleError(`Failed to queue slice for transcription: ${error}`)
        })
      } else {
        this.log(`Skipping slice ${result.slice.index} - already transcribing`)
      }

      this.emitStatsUpdate('memory_change')
    }

    // Clear accumulated data
    this.accumulatedData = new Uint8Array(0)
  }

  /**
   * Check if auto-slice should be triggered based on VAD event and timing
   */
  private async checkAutoSlice(vadEvent: VADEvent, _slice: any): Promise<void> {
    if (!this.options.autoSliceOnSpeechEnd || !this.vadEnabled) {
      return
    }

    // Only trigger on speech_end or silence events
    const shouldTriggerAutoSlice =
      vadEvent.type === 'speech_end' || vadEvent.type === 'silence'

    if (!shouldTriggerAutoSlice) {
      return
    }

    // Get current slice info from SliceManager
    const currentSliceInfo = this.sliceManager.getCurrentSliceInfo()
    const currentSlice = this.sliceManager.getSliceByIndex(
      currentSliceInfo.currentSliceIndex,
    )

    if (!currentSlice) {
      return
    }

    // Calculate current slice duration
    const currentDuration = (Date.now() - currentSlice.startTime) / 1000 // Convert to seconds
    const targetDuration = this.options.audioSliceSec
    const minDuration = this.options.audioMinSec
    const autoSliceThreshold = targetDuration * this.options.autoSliceThreshold

    // Check if conditions are met for auto-slice
    const meetsMinDuration = currentDuration >= minDuration
    const meetsThreshold = currentDuration >= autoSliceThreshold

    if (meetsMinDuration && meetsThreshold) {
      this.log(
        `Auto-slicing on ${vadEvent.type} at ${currentDuration.toFixed(1)}s ` +
          `(min: ${minDuration}s, threshold: ${autoSliceThreshold.toFixed(
            1,
          )}s, target: ${targetDuration}s)`,
      )

      // Force next slice
      await this.nextSlice()
    } else {
      this.log(
        `Auto-slice conditions not met on ${vadEvent.type}: ` +
          `duration=${currentDuration.toFixed(
            1,
          )}s, min=${minDuration}s, threshold=${autoSliceThreshold.toFixed(
            1,
          )}s ` +
          `(minOk=${meetsMinDuration}, thresholdOk=${meetsThreshold})`,
      )
    }
  }

  /**
   * Process VAD for a completed slice
   */
  private async processSliceVAD(slice: any): Promise<void> {
    try {
      // Get audio data from the slice for VAD processing
      const audioData = this.sliceManager.getAudioDataForTranscription(
        slice.index,
      )

      if (!audioData) {
        this.log(
          `No audio data available for VAD processing of slice ${slice.index}`,
        )
        return
      }

      // Convert base64 back to Uint8Array for VAD processing

      // Detect speech in the slice
      const vadEvent = await this.detectSpeech(audioData, slice.index)
      vadEvent.timestamp = Date.now()

      // Emit VAD event
      this.callbacks.onVAD?.(vadEvent)

      // Check if auto-slice should be triggered
      await this.checkAutoSlice(vadEvent, slice)

      // Check if speech was detected and if we should transcribe
      const isSpeech =
        vadEvent.type === 'speech_start' || vadEvent.type === 'speech_continue'

      const isSpeechEnd = vadEvent.type === 'speech_end'

      if (isSpeech) {
        const minDuration = this.options.audioMinSec
        // Check if this is a new speech detection (different from last detected time)
        if (
          vadEvent.lastSpeechDetectedTime !== this.lastSpeechDetectedTime ||
          (vadEvent.lastSpeechDetectedTime - this.lastSpeechDetectedTime) /
            100 >
            minDuration
        ) {
          this.lastSpeechDetectedTime = vadEvent.lastSpeechDetectedTime

          // Check minimum duration requirement
          const speechDuration = slice.data.length / 16000 / 2 // Convert bytes to seconds (16kHz, 16-bit)

          if (speechDuration >= minDuration) {
            this.log(
              `Speech detected in slice ${slice.index}, queueing for transcription`,
            )
            await this.queueSliceForTranscription(slice)
          } else {
            this.log(
              `Speech too short in slice ${
                slice.index
              } (${speechDuration.toFixed(2)}s < ${minDuration}s), skipping`,
            )
          }
        } else {
          this.log(
            `Skipping transcription for slice ${slice.index} - same detection time as last`,
          )
        }
      } else if (isSpeechEnd) {
        this.log(`Speech ended in slice ${slice.index}`)
        // For speech_end events, we might want to queue the slice for transcription
        // to capture the final part of the speech segment
        const speechDuration = slice.data.length / 16000 / 2 // Convert bytes to seconds
        const minDuration = this.options.audioMinSec

        if (speechDuration >= minDuration) {
          this.log(
            `Speech end detected in slice ${slice.index}, queueing final segment for transcription`,
          )
          await this.queueSliceForTranscription(slice)
        } else {
          this.log(
            `Speech end segment too short in slice ${
              slice.index
            } (${speechDuration.toFixed(2)}s < ${minDuration}s), skipping`,
          )
        }
      } else {
        this.log(`No speech detected in slice ${slice.index}`)
      }

      // Emit stats update for VAD change
      this.emitStatsUpdate('vad_change')
    } catch (error: any) {
      this.handleError(
        `VAD processing error for slice ${slice.index}: ${error}`,
      )
    }
  }

  /**
   * Queue a slice for transcription
   */
  private async queueSliceForTranscription(slice: any): Promise<void> {
    try {
      // Get audio data from the slice
      const audioData = this.sliceManager.getAudioDataForTranscription(
        slice.index,
      )

      if (!audioData) {
        this.log(`No audio data available for slice ${slice.index}`)
        return
      }

      // Add to transcription queue
      this.transcriptionQueue.push({
        sliceIndex: slice.index,
        audioData,
      })

      this.log(
        `Queued slice ${slice.index} for transcription (${slice.data.length} samples)`,
      )

      // Emit stats update for queue change
      this.emitStatsUpdate('queue_change')

      await this.processTranscriptionQueue()
    } catch (error: any) {
      this.handleError(`Failed to queue slice for transcription: ${error}`)
    }
  }

  /**
   * Detect speech using VAD context
   */
  private async detectSpeech(
    audioData: Uint8Array,
    sliceIndex: number,
  ): Promise<VADEvent> {
    if (!this.vadContext) {
      // When no VAD context is available, assume speech is always detected
      // but still follow the state machine pattern
      const currentTimestamp = Date.now()

      // Assume speech is always detected when no VAD context
      const vadEventType: VADEvent['type'] =
        this.lastVadState === 'silence' ? 'speech_start' : 'speech_continue'

      // Update VAD state
      this.lastVadState = 'speech'

      return {
        type: vadEventType,
        lastSpeechDetectedTime: 0,
        timestamp: currentTimestamp,
        confidence: 1.0,
        duration: audioData.length / 16000 / 2, // Convert bytes to seconds
        sliceIndex,
      }
    }

    try {
      const audioBuffer = audioData.buffer as SharedArrayBuffer

      // Use VAD context to detect speech segments
      const vadSegments = await this.vadContext.detectSpeechData(
        audioBuffer,
        this.options.vadOptions,
      )

      // Calculate confidence based on speech segments
      let confidence = 0.0
      let lastSpeechDetectedTime = 0
      if (vadSegments && vadSegments.length > 0) {
        // If there are speech segments, calculate average confidence
        const totalTime = vadSegments.reduce(
          (sum, segment) => sum + (segment.t1 - segment.t0),
          0,
        )
        const audioDuration = audioData.length / 16000 / 2 // Convert bytes to seconds
        confidence =
          totalTime > 0 ? Math.min(totalTime / audioDuration, 1.0) : 0.0
        lastSpeechDetectedTime = vadSegments[vadSegments.length - 1]?.t1 || -1
      }

      const threshold = this.options.vadOptions.threshold || 0.5
      const isSpeech = confidence > threshold
      const currentTimestamp = Date.now()

      // Determine VAD event type based on current and previous state
      let vadEventType: VADEvent['type']
      if (isSpeech) {
        vadEventType =
          this.lastVadState === 'silence' ? 'speech_start' : 'speech_continue'
      } else {
        vadEventType = this.lastVadState === 'speech' ? 'speech_end' : 'silence'
      }

      // Update VAD state for next detection
      this.lastVadState = isSpeech ? 'speech' : 'silence'

      return {
        type: vadEventType,
        lastSpeechDetectedTime,
        timestamp: currentTimestamp,
        confidence,
        duration: audioData.length / 16000 / 2, // Convert bytes to seconds
        sliceIndex,
        currentThreshold: threshold,
      }
    } catch (error) {
      this.log(`VAD detection error: ${error}`)

      // Return default silence event on error
      return {
        type: 'silence',
        lastSpeechDetectedTime: 0,
        timestamp: Date.now(),
        confidence: 0.0,
        duration: 0,
        sliceIndex,
      }
    }
  }

  private isProcessingTranscriptionQueue = false

  /**
   * Process the transcription queue
   */
  private async processTranscriptionQueue(): Promise<void> {
    if (this.isProcessingTranscriptionQueue) return

    this.isProcessingTranscriptionQueue = true

    while (this.transcriptionQueue.length > 0) {
      const item = this.transcriptionQueue.shift()
      if (item) {
        // eslint-disable-next-line no-await-in-loop
        await this.processTranscription(item).catch((error) => {
          this.handleError(`Transcription error: ${error}`)
        })
      }
    }

    this.isProcessingTranscriptionQueue = false
  }

  /**
   * Build prompt from initial prompt and previous slices
   */
  private buildPrompt(currentSliceIndex: number): string | undefined {
    const promptParts: string[] = []

    // Add initial prompt if provided
    if (this.options.initialPrompt) {
      promptParts.push(this.options.initialPrompt)
    }

    // Add previous slice results if enabled
    if (this.options.promptPreviousSlices) {
      // Get transcription results from previous slices (up to the current slice)
      const previousResults = Array.from(this.transcriptionResults.entries())
        .filter(([sliceIndex]) => sliceIndex < currentSliceIndex)
        .sort(([a], [b]) => a - b) // Sort by slice index
        .map(([, result]) => result.transcribeEvent.data?.result)
        .filter((result): result is string => Boolean(result)) // Filter out empty results with type guard

      if (previousResults.length > 0) {
        promptParts.push(...previousResults)
      }
    }

    return promptParts.join(' ') || undefined
  }

  /**
   * Process a single transcription
   */
  private async processTranscription(item: {
    sliceIndex: number
    audioData: Uint8Array
  }): Promise<void> {
    if (!this.isActive) {
      return
    }

    this.isTranscribing = true

    // Emit stats update for status change
    this.emitStatsUpdate('status_change')

    const startTime = Date.now()

    try {
      // Build prompt from initial prompt and previous slices
      const prompt = this.buildPrompt(item.sliceIndex)

      const audioBuffer = item.audioData.buffer as SharedArrayBuffer
      const { promise } = this.whisperContext.transcribeData(audioBuffer, {
        ...this.options.transcribeOptions,
        prompt, // Include the constructed prompt
        onProgress: undefined, // Disable progress for realtime
      })

      const result = await promise
      const endTime = Date.now()

      // Create transcribe event
      const transcribeEvent: TranscribeEvent = {
        type: 'transcribe',
        sliceIndex: item.sliceIndex,
        data: result,
        isCapturing: this.audioStream.isRecording(),
        processTime: endTime - startTime,
        recordingTime: (result.segments?.length || 0) * 1000, // Estimate
        memoryUsage: this.sliceManager.getMemoryUsage(),
      }

      // Emit transcribe event
      this.callbacks.onTranscribe?.(transcribeEvent)

      // Save transcription results
      const slice = this.sliceManager.getSliceByIndex(item.sliceIndex)
      if (slice) {
        this.transcriptionResults.set(item.sliceIndex, {
          slice: {
            // Don't keep data in the slice
            index: slice.index,
            sampleCount: slice.sampleCount,
            startTime: slice.startTime,
            endTime: slice.endTime,
            isProcessed: slice.isProcessed,
            isReleased: slice.isReleased,
          },
          transcribeEvent,
        })
      }

      // Emit stats update for memory/slice changes
      this.emitStatsUpdate('memory_change')

      this.log(
        `Transcribed speech segment ${item.sliceIndex}: "${result.result}"`,
      )
    } catch (error) {
      this.handleError(
        `Transcription failed for speech segment ${item.sliceIndex}: ${error}`,
      )
    } finally {
      // Check if we should continue processing queue
      if (this.transcriptionQueue.length > 0) {
        await this.processTranscriptionQueue()
      } else {
        this.isTranscribing = false
      }

      // Emit stats update for queue change
      this.emitStatsUpdate('queue_change')
    }
  }

  /**
   * Handle audio status changes
   */
  private handleAudioStatusChange(isRecording: boolean): void {
    this.log(`Audio recording: ${isRecording ? 'started' : 'stopped'}`)
  }

  /**
   * Handle errors from components
   */
  private handleError(error: string): void {
    this.log(`Error: ${error}`)
    this.callbacks.onError?.(error)
  }

  /**
   * Update callbacks
   */
  updateCallbacks(callbacks: Partial<RealtimeTranscriberCallbacks>): void {
    this.callbacks = { ...this.callbacks, ...callbacks }
  }

  /**
   * Update VAD options dynamically
   */
  updateVadOptions(options: Partial<VadOptions>): void {
    this.options.vadOptions = { ...this.options.vadOptions, ...options }
  }

  /**
   * Update auto-slice options dynamically
   */
  updateAutoSliceOptions(options: {
    autoSliceOnSpeechEnd?: boolean
    autoSliceThreshold?: number
  }): void {
    if (options.autoSliceOnSpeechEnd !== undefined) {
      this.options.autoSliceOnSpeechEnd = options.autoSliceOnSpeechEnd
    }
    if (options.autoSliceThreshold !== undefined) {
      this.options.autoSliceThreshold = options.autoSliceThreshold
    }
    this.log(
      `Auto-slice options updated: enabled=${this.options.autoSliceOnSpeechEnd}, threshold=${this.options.autoSliceThreshold}`,
    )
  }

  /**
   * Get current statistics
   */
  getStatistics() {
    return {
      isActive: this.isActive,
      isTranscribing: this.isTranscribing,
      vadEnabled: this.vadEnabled,
      queueLength: this.transcriptionQueue.length,
      audioStats: {
        isRecording: this.audioStream.isRecording(),
        accumulatedSamples: this.accumulatedData.length,
      },
      vadStats: this.vadEnabled
        ? {
            enabled: true,
            contextAvailable: !!this.vadContext,
            lastSpeechDetectedTime: this.lastSpeechDetectedTime,
          }
        : null,
      sliceStats: this.sliceManager.getCurrentSliceInfo(),
      autoSliceConfig: {
        enabled: this.options.autoSliceOnSpeechEnd,
        threshold: this.options.autoSliceThreshold,
        targetDuration: this.options.audioSliceSec,
        minDuration: this.options.audioMinSec,
      },
    }
  }

  /**
   * Get all transcription results
   */
  getTranscriptionResults(): Array<{
    slice: AudioSliceNoData
    transcribeEvent: TranscribeEvent
  }> {
    return Array.from(this.transcriptionResults.values())
  }

  /**
   * Force move to the next slice, finalizing the current one regardless of capacity
   */
  async nextSlice(): Promise<void> {
    if (!this.isActive) {
      this.log('Cannot force next slice - transcriber is not active')
      return
    }

    // Check if there are pending transcriptions or currently transcribing
    if (this.isTranscribing || this.transcriptionQueue.length > 0) {
      this.log(
        'Waiting for pending transcriptions to complete before forcing next slice...',
      )

      // Wait for current transcription queue to be processed
      await this.processTranscriptionQueue()
    }

    const result = this.sliceManager.forceNextSlice()

    if (result.slice) {
      this.log(
        `Forced slice ${result.slice.index} ready (${result.slice.data.length} bytes)`,
      )

      // Process VAD for the slice if enabled
      if (!this.isTranscribing && this.vadEnabled) {
        this.processSliceVAD(result.slice).catch((error: any) => {
          this.handleError(`VAD processing error: ${error}`)
        })
      } else if (!this.isTranscribing) {
        // If VAD is disabled, transcribe slices as they become ready
        this.queueSliceForTranscription(result.slice).catch((error: any) => {
          this.handleError(`Failed to queue slice for transcription: ${error}`)
        })
      } else {
        this.log(`Skipping slice ${result.slice.index} - already transcribing`)
      }

      this.emitStatsUpdate('memory_change')
    } else {
      this.log('Forced next slice but no slice data to process')
    }
  }

  /**
   * Reset all components
   */
  reset(): void {
    this.sliceManager.reset()
    this.transcriptionQueue = []
    this.isTranscribing = false
    this.accumulatedData = new Uint8Array(0)

    // Reset simplified VAD state
    this.lastSpeechDetectedTime = -1
    this.lastVadState = 'silence'

    // Reset stats snapshot for clean start
    this.lastStatsSnapshot = null

    // Cancel WAV file writing if in progress
    if (this.wavFileWriter) {
      this.wavFileWriter.cancel().catch((error) => {
        console.warn('Failed to cancel WAV file writing:', error)
      })
      this.wavFileWriter = null
    }

    // Clear transcription results
    this.transcriptionResults.clear()
  }

  /**
   * Release all resources
   */
  async release(): Promise<void> {
    if (this.isActive) {
      await this.stop()
    }

    await this.audioStream.release()
    await this.wavFileWriter?.finalize()
    this.vadContext = undefined
  }

  /**
   * Emit stats update event if stats have changed significantly
   */
  private emitStatsUpdate(eventType: StatsEvent['type']): void {
    const currentStats = this.getStatistics()

    // Check if stats have changed significantly
    if (
      !this.lastStatsSnapshot ||
      RealtimeTranscriber.shouldEmitStatsUpdate(
        currentStats,
        this.lastStatsSnapshot,
      )
    ) {
      const statsEvent: StatsEvent = {
        timestamp: Date.now(),
        type: eventType,
        data: currentStats,
      }

      this.callbacks.onStatsUpdate?.(statsEvent)
      this.lastStatsSnapshot = { ...currentStats }
    }
  }

  /**
   * Determine if stats update should be emitted
   */
  private static shouldEmitStatsUpdate(current: any, previous: any): boolean {
    // Always emit on status changes
    if (
      current.isActive !== previous.isActive ||
      current.isTranscribing !== previous.isTranscribing
    ) {
      return true
    }

    // Emit on queue length changes
    if (current.queueLength !== previous.queueLength) {
      return true
    }

    // Emit on significant memory changes (>10% or >5MB)
    const currentMemory = current.sliceStats?.memoryUsage?.estimatedMB || 0
    const previousMemory = previous.sliceStats?.memoryUsage?.estimatedMB || 0
    const memoryDiff = Math.abs(currentMemory - previousMemory)

    if (
      memoryDiff > 5 ||
      (previousMemory > 0 && memoryDiff / previousMemory > 0.1)
    ) {
      return true
    }

    return false
  }

  /**
   * Debug logging
   */
  private log(message: string): void {
    console.log(`[RealtimeTranscriber] ${message}`)
  }
}
