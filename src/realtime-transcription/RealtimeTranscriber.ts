/* eslint-disable class-methods-use-this */
import type { VadOptions } from '../index'
import { SliceManager } from './SliceManager'
import { WavFileWriter, WavFileWriterFs } from '../utils/WavFileWriter'
import type {
  RealtimeOptions,
  RealtimeTranscribeEvent,
  RealtimeVadEvent,
  RealtimeTranscriberCallbacks,
  RealtimeStatsEvent,
  RealtimeTranscriberDependencies,
  AudioStreamData,
  AudioSliceNoData,
  AudioStreamInterface,
  AudioStreamConfig,
  WhisperContextLike,
  RealtimeVadContextLike,
} from './types'

const SILENCE_SEGMENT_REGEX = /\[(\s*\w+\s*)]/i

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
  private whisperContext: WhisperContextLike

  private vadContext?: RealtimeVadContextLike

  private audioStream: AudioStreamInterface

  private fs?: WavFileWriterFs

  private sliceManager: SliceManager

  private callbacks: RealtimeTranscriberCallbacks = {}

  private options: {
    audioSliceSec: number
    audioMinSec: number
    maxSlicesInMemory: number
    transcribeOptions: any
    initialPrompt?: string
    promptPreviousSlices: boolean
    audioOutputPath?: string
    audioStreamConfig?: AudioStreamConfig
    realtimeProcessingPauseMs: number
    initRealtimeAfterMs: number
    logger: (message: string) => void
  }

  private isActive = false

  private isTranscribing = false

  private vadEnabled = false

  private isSpeechActive = false

  private transcriptionQueue: Array<{
    sliceIndex: number
    audioData: Uint8Array
    isFinal?: boolean
  }> = []

  private wavFileWriter: WavFileWriter | null = null

  // Simplified VAD state management
  private lastSpeechDetectedTime = 0

  // Track last stats to emit only when changed
  private lastStatsSnapshot: any = null

  // Track last realtime transcription time for throttling
  private lastRealtimeTranscriptionTime = 0

  // Store transcription results by slice index
  private transcriptionResults: Map<
    number,
    { slice: AudioSliceNoData; transcribeEvent: RealtimeTranscribeEvent }
  > = new Map()

  // Store VAD events by slice index for inclusion in transcribe events
  private vadEvents: Map<number, RealtimeVadEvent> = new Map()

  // Track active async operations
  private activeTranscriptions: Set<{ promise: Promise<any> }> = new Set()

  constructor(
    dependencies: RealtimeTranscriberDependencies,
    options: RealtimeOptions = {},
    callbacks: RealtimeTranscriberCallbacks = {},
  ) {
    this.whisperContext = dependencies.whisperContext
    this.vadContext = dependencies.vadContext
    this.audioStream = dependencies.audioStream
    this.fs = dependencies.fs
    this.callbacks = callbacks

    // Set default options with proper types
    this.options = {
      audioSliceSec: options.audioSliceSec || 30,
      audioMinSec: options.audioMinSec || 1,
      maxSlicesInMemory: options.maxSlicesInMemory || 3,
      transcribeOptions: options.transcribeOptions || {},
      initialPrompt: options.initialPrompt,
      promptPreviousSlices: options.promptPreviousSlices ?? true,
      audioOutputPath: options.audioOutputPath,
      realtimeProcessingPauseMs: options.realtimeProcessingPauseMs || 200,
      initRealtimeAfterMs: options.initRealtimeAfterMs || 200,
      logger: options.logger || (() => { }),
    }

    // Enable VAD if context is provided
    this.vadEnabled = !!this.vadContext

    if (this.vadContext) {
      this.vadContext.onSpeechStart(this.handleSpeechDetected.bind(this))
      this.vadContext.onSpeechContinue(this.handleSpeechContinue.bind(this))
      this.vadContext.onSpeechEnd(this.handleSpeechEnded.bind(this))
      this.vadContext.onError(this.handleError.bind(this))
    }

    // Initialize managers
    this.sliceManager = new SliceManager(
      this.options.audioSliceSec,
      this.options.maxSlicesInMemory,
    )

    // Set up audio stream callbacks
    this.audioStream.onData(this.handleAudioData.bind(this))
    this.audioStream.onError(this.handleError.bind(this))
    this.audioStream.onStatusChange(this.handleAudioStatusChange.bind(this))
    this.audioStream.onEnd?.(this.handleAudioEnd.bind(this))
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
      if (this.fs && this.options.audioOutputPath) {
        this.wavFileWriter = new WavFileWriter(
          this.fs,
          this.options.audioOutputPath,
          {
            sampleRate: this.options.audioStreamConfig?.sampleRate || 16000,
            channels: this.options.audioStreamConfig?.channels || 1,
            bitsPerSample: this.options.audioStreamConfig?.bitsPerSample || 16,
          },
        )
        await this.wavFileWriter.initialize()
      }

      // Start audio recording
      await this.audioStream.initialize({
        sampleRate: this.options.audioStreamConfig?.sampleRate || 16000,
        channels: this.options.audioStreamConfig?.channels || 1,
        bitsPerSample: this.options.audioStreamConfig?.bitsPerSample || 16,
        audioSource: this.options.audioStreamConfig?.audioSource || 6,
        bufferSize: this.options.audioStreamConfig?.bufferSize || 16 * 1024,
      })
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

      // Stop audio recording first to stop new data coming in
      await this.audioStream.stop()

      // Process any remaining queued transcriptions
      await this.processTranscriptionQueue()

      // Wait for all active transcriptions to complete
      await Promise.allSettled([...this.activeTranscriptions].map(t => t.promise))
      this.activeTranscriptions.clear()

      // Reset VAD context (waits for its internal active promises)
      if (this.vadContext) {
        await this.vadContext.reset()
      }

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
   * Handle incoming audio data
   */
  private handleAudioData(streamData: AudioStreamData): void {
    if (!this.isActive) return

    this.processAudioChunk(streamData.data).catch((error) => {
      this.handleError(`Audio processing error: ${error}`)
    })

    // Write to WAV file if enabled
    if (this.wavFileWriter) {
      this.wavFileWriter.appendAudioData(streamData.data).catch((error) => {
        this.log(`Failed to write audio to WAV file: ${error}`)
      })
    }
  }

  /**
   * Process audio chunk through the VAD pipeline
   */
  private async processAudioChunk(data: Uint8Array): Promise<void> {
    // Push directly to VAD context
    if (this.vadContext) {
      // Check pre-VAD filter if exists (optional callback)
      if (this.callbacks.onBeginVad) {
        const { sampleRate = 16000 } = this.options.audioStreamConfig || {}
        const duration = (data.length / 2) / (sampleRate / 1000) // ms

        const shouldContinue = await this.callbacks.onBeginVad({
          audioData: data,
          sliceIndex: -1, // No slice index yet for raw chunks
          duration,
        })

        if (!shouldContinue) {
          // User cancelled VAD for this chunk
          return
        }
      }

      this.vadContext.processAudio(data)
    } else {
      // Fallback: If no VAD context, treat everything as speech/audio to be processed
      this.sliceManager.addAudioData(data)
      this.triggerTranscription(false)
    }
  }

  // --- VAD Handlers ---

  private async handleSpeechDetected(confidence: number, data: Uint8Array): Promise<void> {
    if (!this.isActive) return
    if (!this.isSpeechActive) {
      // Speech Start
      this.isSpeechActive = true
      this.log('VAD: Speech Start detected')
      this.lastSpeechDetectedTime = Date.now()
      this.emitVadEvent('speech_start', confidence)

      this.sliceManager.addAudioData(data)
      this.triggerTranscription(false)
    }
  }

  private async handleSpeechContinue(confidence: number, data: Uint8Array): Promise<void> {
    if (!this.isActive || !this.isSpeechActive) return
    this.emitVadEvent('speech_continue', confidence)
    this.sliceManager.addAudioData(data)
    this.triggerTranscription(false)
  }

  private async handleSpeechEnded(confidence: number): Promise<void> {
    if (!this.isActive) return

    this.isSpeechActive = false
    this.emitVadEvent('speech_end', confidence)
    await this.nextSlice()
  }

  /**
   * Trigger transcription for the current slice
   */
  private triggerTranscription(isFinal: boolean): void {
    const sliceInfo = this.sliceManager.getCurrentSliceInfo()
    const slice = this.sliceManager.getSliceByIndex(sliceInfo.currentSliceIndex)

    if (!slice || slice.sampleCount === 0) return

    // Queue transcription
    const audioData = this.sliceManager.getAudioDataForTranscription(slice.index)
    if (audioData) {
      // Throttling logic for realtime (non-final) transcriptions
      if (!isFinal) {
        const { sampleRate = 16000 } = this.options.audioStreamConfig || {}
        const durationMs = (audioData.length / 2) / (sampleRate / 1000)
        const now = Date.now()

        // 1. Initial wait: Don't transcribe if slice is too short (unless it's final, which checks above handle)
        if (durationMs < this.options.initRealtimeAfterMs) {
          return
        }

        // 2. Throttling: Don't transcribe if too soon after last update
        if (now - this.lastRealtimeTranscriptionTime < this.options.realtimeProcessingPauseMs) {
          return
        }

        this.lastRealtimeTranscriptionTime = now
      }

      this.transcriptionQueue.push({
        sliceIndex: slice.index,
        audioData,
        isFinal // Pass flag to processTranscription (need update)
      })
      this.processTranscriptionQueue().catch(e => this.handleError(e))
    }
  }

  private emitVadEvent(type: RealtimeVadEvent['type'], confidence: number): void {
    const sliceInfo = this.sliceManager.getCurrentSliceInfo()
    const event: RealtimeVadEvent = {
      type,
      timestamp: Date.now(),
      sliceIndex: sliceInfo.currentSliceIndex,
      confidence,
      lastSpeechDetectedTime: this.lastSpeechDetectedTime,
      duration: this.sliceManager.getSliceByIndex(sliceInfo.currentSliceIndex)?.data.length ? this.sliceManager.getSliceByIndex(sliceInfo.currentSliceIndex)!.data.length / 32000 : 0
    }
    this.vadEvents.set(sliceInfo.currentSliceIndex, event)
    this.callbacks.onVad?.(event)
  }

  private isProcessingTranscriptionQueue = false

  private processingPromise: Promise<void> | null = null

  /**
   * Process the transcription queue
   */
  private async processTranscriptionQueue(): Promise<void> {
    if (this.isProcessingTranscriptionQueue && this.processingPromise) {
      return this.processingPromise
    }

    this.isProcessingTranscriptionQueue = true

    this.processingPromise = (async () => {
      while (this.transcriptionQueue.length > 0) {
        const item = this.transcriptionQueue.shift() // shift() modifies the array
        if (item) {
          // eslint-disable-next-line no-await-in-loop
          await this.processTranscription(item).catch((error) => {
            this.handleError(`Transcription error: ${error}`)
          })
        }
      }
      this.isProcessingTranscriptionQueue = false
      this.processingPromise = null
    })()

    return this.processingPromise
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
    isFinal?: boolean
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

      const audioBuffer = item.audioData.buffer as ArrayBuffer
      const transcribeRequest = this.whisperContext.transcribeData(audioBuffer, {
        ...this.options.transcribeOptions,
        prompt, // Include the constructed prompt
        onProgress: undefined, // Disable progress for realtime
      })

      // Track active transcription
      this.activeTranscriptions.add(transcribeRequest)

      let result
      try {
        result = await transcribeRequest.promise
      } finally {
        this.activeTranscriptions.delete(transcribeRequest)
      }

      // Check if stopped during transcription
      if (!this.isActive) return

      const endTime = Date.now()

      // Normalize result and segments, remove "[ silence ]" or "[BLANK]"
      result.result = result.result.replace(SILENCE_SEGMENT_REGEX, '').trim()

      const slice = this.sliceManager.getSliceByIndex(item.sliceIndex)
      if (!slice) {
        this.log(`Slice not found for index ${item.sliceIndex}, skipping transcription processing.`)
        return
      }

      // Check if user wants to filter this transcription
      if (this.callbacks.onBeginTranscribe) {
        const { sampleRate = 16000 } = this.options.audioStreamConfig || {}
        const duration = (item.audioData.length / 2) / (sampleRate / 1000) // ms

        const shouldContinue = await this.callbacks.onBeginTranscribe({
          audioData: item.audioData,
          sliceIndex: slice.index,
          duration,
          vadEvent: this.vadEvents.get(item.sliceIndex)
        })

        if (!shouldContinue) {
          this.log(`Transcription filtered by onBeginTranscribe for slice ${slice.index}`)
          return
        }
      }

      // Create new transcription event
      const { sampleRate = 16000 } = this.options.audioStreamConfig || {}
      const transcribeEvent: RealtimeTranscribeEvent = {
        type: 'transcribe',
        sliceIndex: item.sliceIndex,
        data: result,
        isCapturing: this.audioStream.isRecording(),
        processTime: endTime - startTime,
        recordingTime: item.audioData.length / (sampleRate / 1000) / 2, // ms,
        memoryUsage: this.sliceManager.getMemoryUsage(),
        vadEvent: this.vadEvents.get(item.sliceIndex),
      }

      // if the current result is invalid, use the previous result
      const previousTranscribe = this.transcriptionResults.get(item.sliceIndex)
        ?.transcribeEvent
      if (previousTranscribe && result.result.trim() === '.') {
        transcribeEvent.data = previousTranscribe.data
      }

      // Save transcription results
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

      // Emit transcribe event
      this.callbacks.onTranscribe?.(transcribeEvent)

      // Feed result to stabilizer for realtime updates
      // Only stabilize final results (speech_end) to match legacy behavior
      const resultText = result.result?.trim() || ''
      if (item.isFinal) {
        this.callbacks.onSliceTranscriptionStabilized?.(resultText)
        this.vadEvents.delete(item.sliceIndex)
      }

      // Emit stats update for memory/slice changes
      this.emitStatsUpdate('memory_change')

      this.log(
        `Transcribed speech segment ${item.sliceIndex} (Final=${!!item.isFinal}): "${result.result}"`,
      )
    } catch (error) {
      // ... error handling ...
      this.handleError(`Transcription error: ${error}`)
    } finally {
      if (this.transcriptionQueue.length === 0) {
        this.isTranscribing = false
      }
    }
  }

  /**
   * Handle audio stream end
   */
  private async handleAudioEnd(): Promise<void> {
    this.log('Audio stream ended')

    if (this.vadContext) {
      await this.vadContext.flush()
    }

    // If speech is still active after flush, force end it
    if (this.isSpeechActive) {
      this.log('Speech still active after stream end, forcing speech end')
      await this.handleSpeechEnded(1.0)
    }

    // Ensure last slice is processed if it has data
    await this.nextSlice()
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
   * Update VAD options dynamically (delegates to VAD context)
   */
  updateVadOptions(options: Partial<VadOptions>): void {
    if (this.vadContext) {
      this.vadContext.updateOptions(options)
    }
  }

  /**
   * Get current statistics
   */
  getStatistics() {
    return {
      isActive: this.isActive,
      isTranscribing: this.isTranscribing,
      vadEnabled: this.vadEnabled,
      audioStats: {
        isRecording: this.audioStream.isRecording(),
        accumulatedSamples: this.sliceManager.getCurrentSliceInfo().memoryUsage.totalSamples,
      },
      vadStats: this.vadEnabled
        ? {
          enabled: true,
          contextAvailable: !!this.vadContext,
          lastSpeechDetectedTime: this.lastSpeechDetectedTime,
        }
        : null,
      sliceStats: this.sliceManager.getCurrentSliceInfo(),
    }
  }

  /**
   * Get all transcription results
   */
  getTranscriptionResults(): Array<{
    slice: AudioSliceNoData
    transcribeEvent: RealtimeTranscribeEvent
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

    // Emit start event to indicate slice processing has started
    const startEvent: RealtimeTranscribeEvent = {
      type: 'start',
      sliceIndex: -1, // Use -1 to indicate forced slice
      data: undefined,
      isCapturing: this.audioStream.isRecording(),
      processTime: 0,
      recordingTime: 0,
      memoryUsage: this.sliceManager.getMemoryUsage(),
    }

    this.callbacks.onTranscribe?.(startEvent)

    // Check if there are pending transcriptions or currently transcribing
    // We don't need to wait explicitly because the queue handles serialization
    if (this.isTranscribing || this.transcriptionQueue.length > 0) {
      this.log(
        'Queuing forced slice after pending transcriptions...',
      )
    }

    const result = this.sliceManager.forceNextSlice()

    if (result.slice) {
      this.log(
        `Forced slice ${result.slice.index} ready (${result.slice.data.length} bytes)`,
      )

      // Queue for transcription (Final)
      if (result.slice.data.length > 0) {
        this.transcriptionQueue.push({
          sliceIndex: result.slice.index,
          audioData: result.slice.data,
          isFinal: true
        })
        this.processTranscriptionQueue().catch((error) => {
          this.handleError(`Failed to process forced slice: ${error}`)
        })
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

    // Reset VAD state
    this.lastSpeechDetectedTime = -1

    // Reset stats snapshot for clean start
    this.lastStatsSnapshot = null

    this.lastRealtimeTranscriptionTime = 0

    // Cancel WAV file writing if in progress
    if (this.wavFileWriter) {
      this.wavFileWriter.cancel().catch((error) => {
        this.log(`Failed to cancel WAV file writing: ${error}`)
      })
      this.wavFileWriter = null
    }

    // Clear transcription results
    this.transcriptionResults.clear()

    // Clear VAD events
    this.vadEvents.clear()

    this.isSpeechActive = !this.vadContext

    // vadContext is reset in stop(), but if we just call reset() directly:
    if (this.vadContext) {
      this.vadContext.reset().catch(e => this.log(`VAD reset error: ${e}`))
    }
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

    // reset/clear VAD context
    if (this.vadContext) {
      await this.vadContext.reset()
    }
    this.vadContext = undefined
  }

  /**
   * Emit stats update event if stats have changed significantly
   */
  private emitStatsUpdate(eventType: RealtimeStatsEvent['type']): void {
    const currentStats = this.getStatistics()

    // Check if stats have changed significantly
    if (
      !this.lastStatsSnapshot ||
      RealtimeTranscriber.shouldEmitStatsUpdate(
        currentStats,
        this.lastStatsSnapshot,
      )
    ) {
      const statsEvent: RealtimeStatsEvent = {
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
   * Logger function
   */
  private log(message: string): void {
    this.options.logger(`[RealtimeTranscriber] ${message}`)
  }
}
