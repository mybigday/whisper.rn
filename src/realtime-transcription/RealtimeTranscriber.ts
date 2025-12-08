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
  WhisperVadContextLike,
} from './types'
import { VAD_PRESETS } from './types'

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

  private vadContext?: WhisperVadContextLike

  private audioStream: AudioStreamInterface

  private fs?: WavFileWriterFs

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
    audioOutputPath?: string
    audioStreamConfig?: AudioStreamConfig
    logger: (message: string) => void
    // VAD optimization options for low-end CPU
    vadThrottleMs: number // Minimum time between VAD calls (ms)
    vadSkipRatio: number // Skip every Nth slice (0 = no skipping)
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

  // VAD throttling for low-end CPU optimization
  private isProcessingVAD = false

  private lastVadProcessTime = 0

  private vadProcessingQueue: any[] = []

  private skippedVadCount = 0

  // Store transcription results by slice index
  private transcriptionResults: Map<
    number,
    { slice: AudioSliceNoData; transcribeEvent: RealtimeTranscribeEvent }
  > = new Map()

  // Store VAD events by slice index for inclusion in transcribe events
  private vadEvents: Map<number, RealtimeVadEvent> = new Map()

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
      vadOptions: options.vadOptions || VAD_PRESETS.default,
      vadPreset: options.vadPreset,
      autoSliceOnSpeechEnd: options.autoSliceOnSpeechEnd || true,
      autoSliceThreshold: options.autoSliceThreshold || 0.5,
      transcribeOptions: options.transcribeOptions || {},
      initialPrompt: options.initialPrompt,
      promptPreviousSlices: options.promptPreviousSlices ?? true,
      audioOutputPath: options.audioOutputPath,
      logger: options.logger || (() => {}),
      // VAD optimization options for low-end CPU
      vadThrottleMs: options.vadThrottleMs ?? 1500, // Minimum time between VAD calls (ms)
      vadSkipRatio: options.vadSkipRatio ?? 0, // Skip every Nth slice (0 = no skipping)
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
          this.log(`Failed to write audio to WAV file: ${error}`)
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

      // Process VAD for the slice if enabled (with throttling for low-end CPU)
      if (!this.isTranscribing && this.vadEnabled) {
        this.queueVADProcessing(result.slice)
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
  private async checkAutoSlice(
    vadEvent: RealtimeVadEvent,
    _slice: any,
  ): Promise<void> {
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
   * Queue VAD processing with throttling for low-end CPU systems
   * This prevents VAD from blocking the audio pipeline
   */
  private queueVADProcessing(slice: any): void {
    // Check if we should skip this slice based on skip ratio
    if (this.options.vadSkipRatio > 0) {
      const shouldSkip = slice.index % (this.options.vadSkipRatio + 1) !== 0
      if (shouldSkip) {
        this.skippedVadCount += 1
        this.log(
          `Skipping VAD for slice ${slice.index} (skip ratio: ${this.options.vadSkipRatio})`,
        )
        // Still queue for transcription if VAD would have approved
        this.queueSliceForTranscription(slice).catch((error: any) => {
          this.handleError(`Failed to queue skipped slice for transcription: ${error}`)
        })
        return
      }
    }

    // Check throttling - don't process if we recently processed VAD
    const now = Date.now()
    const timeSinceLastVad = now - this.lastVadProcessTime

    if (this.isProcessingVAD) {
      // VAD is already running, queue this slice
      this.vadProcessingQueue.push(slice)
      this.log(
        `VAD busy, queued slice ${slice.index} (queue size: ${this.vadProcessingQueue.length})`,
      )
      return
    }

    if (timeSinceLastVad < this.options.vadThrottleMs) {
      // Too soon since last VAD, queue it
      this.vadProcessingQueue.push(slice)
      this.log(
        `VAD throttled, queued slice ${slice.index} (will process in ${
          this.options.vadThrottleMs - timeSinceLastVad
        }ms)`,
      )
      // Schedule processing after throttle period
      setTimeout(() => {
        this.processVADQueue()
      }, this.options.vadThrottleMs - timeSinceLastVad)
      return
    }

    this.processSliceVADThrottled(slice)
  }

  /**
   * Process the VAD queue
   */
  private processVADQueue(): void {
    if (this.isProcessingVAD || this.vadProcessingQueue.length === 0) {
      return
    }

    // Get the most recent slice from queue (discard older ones for real-time performance)
    const slice = this.vadProcessingQueue.pop()
    this.vadProcessingQueue = [] // Clear queue, we only care about latest

    if (slice) {
      this.log(`Processing queued VAD for slice ${slice.index}`)
      this.processSliceVADThrottled(slice)
    }
  }

  /**
   * Throttled wrapper for processSliceVAD
   */
  private async processSliceVADThrottled(slice: any): Promise<void> {
    if (this.isProcessingVAD) {
      // Already processing, re-queue
      this.vadProcessingQueue.push(slice)
      return
    }

    this.isProcessingVAD = true
    this.lastVadProcessTime = Date.now()

    try {
      await this.processSliceVAD(slice)
    } catch (error) {
      this.handleError(`VAD processing error: ${error}`)
    } finally {
      this.isProcessingVAD = false

      // Process next item in queue if available
      if (this.vadProcessingQueue.length > 0) {
        // Schedule next processing with a small delay to yield to event loop
        setTimeout(() => {
          this.processVADQueue()
        }, 50) // 50ms delay between VAD processings
      }
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

      // Check if user callback allows VAD processing
      if (this.callbacks.onBeginVad) {
        const {
          sampleRate = 16000,
          channels = 1,
        } = this.options.audioStreamConfig || {}
        const duration = audioData.length / sampleRate / channels * 1000 // Convert to milliseconds
        const shouldProcessVad =
          (await this.callbacks.onBeginVad({
            sliceIndex: slice.index,
            audioData,
            duration,
          })) ?? true

        if (!shouldProcessVad) {
          this.log(`User callback declined VAD processing for slice ${slice.index}`)
          return
        }
      }

      // Detect speech in the slice
      const vadEvent = await this.detectSpeech(audioData, slice.index)
      vadEvent.timestamp = Date.now()

      // Store VAD event for inclusion in transcribe event
      this.vadEvents.set(slice.index, vadEvent)

      // Emit VAD event
      this.callbacks.onVad?.(vadEvent)

      // Check if auto-slice should be triggered
      await this.checkAutoSlice(vadEvent, slice)

      // Check if speech was detected and if we should transcribe
      const isSpeech =
        vadEvent.type === 'speech_start' || vadEvent.type === 'speech_continue'

      const isSpeechEnd = vadEvent.type === 'speech_end'

      if (isSpeech) {
        const minDuration = this.options.audioMinSec
        // Check minimum duration requirement
        const speechDuration = slice.data.length / 16000 / 2 // Convert bytes to seconds (16kHz, 16-bit)

        if (speechDuration >= minDuration) {
          this.log(
            `Speech detected in slice ${slice.index}, queueing for transcription`,
          )
          await this.queueSliceForTranscription(slice)
        } else {
          this.log(
            `Speech too short in slice ${slice.index} (${speechDuration.toFixed(
              2,
            )}s < ${minDuration}s), skipping`,
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

      if (this.callbacks.onBeginTranscribe) {
        const shouldTranscribe =
          (await this.callbacks.onBeginTranscribe({
            sliceIndex: slice.index,
            audioData,
            duration: (slice.data.length / 16000 / 2) * 1000, // Convert to milliseconds
            vadEvent: this.vadEvents.get(slice.index),
          })) ?? true

        if (!shouldTranscribe) {
          this.log(
            `User callback declined transcription for slice ${slice.index}`,
          )
          return
        }
      }

      // Add to transcription queue
      this.transcriptionQueue.unshift({
        sliceIndex: slice.index,
        audioData,
      })

      this.log(
        `Queued slice ${slice.index} for transcription (${slice.data.length} samples)`,
      )

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
  ): Promise<RealtimeVadEvent> {
    if (!this.vadContext) {
      // When no VAD context is available, assume speech is always detected
      // but still follow the state machine pattern
      const currentTimestamp = Date.now()

      // Assume speech is always detected when no VAD context
      const vadEventType: RealtimeVadEvent['type'] =
        this.lastVadState === 'silence' ? 'speech_start' : 'speech_continue'

      // Update VAD state
      this.lastVadState = 'speech'

      const { sampleRate = 16000 } = this.options.audioStreamConfig || {}
      return {
        type: vadEventType,
        lastSpeechDetectedTime: 0,
        timestamp: currentTimestamp,
        confidence: 1.0,
        duration: audioData.length / sampleRate / 2, // Convert bytes to seconds
        sliceIndex,
      }
    }

    try {
      const audioBuffer = audioData.buffer as ArrayBuffer

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
      let isSpeech = confidence > threshold
      const currentTimestamp = Date.now()

      // Determine VAD event type based on current and previous state
      let vadEventType: RealtimeVadEvent['type']
      if (isSpeech) {
        vadEventType =
          this.lastVadState === 'silence' ? 'speech_start' : 'speech_continue'

        const minDuration = this.options.audioMinSec
        // Check if this is a new speech detection (different from last detected time)
        if (
          lastSpeechDetectedTime === this.lastSpeechDetectedTime ||
          (lastSpeechDetectedTime - this.lastSpeechDetectedTime) / 100 <
            minDuration
        ) {
          if (this.lastVadState === 'silence') vadEventType = 'silence'
          if (this.lastVadState === 'speech') vadEventType = 'speech_end'
          isSpeech = false
          confidence = 0.0
        }
        this.lastSpeechDetectedTime = lastSpeechDetectedTime
      } else {
        vadEventType = this.lastVadState === 'speech' ? 'speech_end' : 'silence'
      }

      // Update VAD state for next detection
      this.lastVadState = isSpeech ? 'speech' : 'silence'

      const { sampleRate = 16000 } = this.options.audioStreamConfig || {}
      return {
        type: vadEventType,
        lastSpeechDetectedTime,
        timestamp: currentTimestamp,
        confidence,
        duration: audioData.length / sampleRate / 2, // Convert bytes to seconds
        sliceIndex,
        currentThreshold: threshold,
      }
    } catch (error) {
      this.log(`VAD detection error: ${error}`)
      // Re-throw the error so it can be handled by the caller
      throw error
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
      this.transcriptionQueue = [] // Old items are not needed anymore
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

      const audioBuffer = item.audioData.buffer as ArrayBuffer
      const { promise } = this.whisperContext.transcribeData(audioBuffer, {
        ...this.options.transcribeOptions,
        prompt, // Include the constructed prompt
        onProgress: undefined, // Disable progress for realtime
      })

      const result = await promise
      const endTime = Date.now()

      // Normalize result and segments, remove "[ silence ]" or "[BLANK]"
      result.result = result.result.replace(SILENCE_SEGMENT_REGEX, '').trim()

      // Create transcribe event
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

      // Emit transcribe event
      this.callbacks.onTranscribe?.(transcribeEvent)

      this.vadEvents.delete(item.sliceIndex)

      // Emit stats update for memory/slice changes
      this.emitStatsUpdate('memory_change')

      this.log(
        `Transcribed speech segment ${item.sliceIndex}: "${result.result}"`,
      )
    } catch (error) {
      // Emit error event to transcribe callback
      const errorEvent: RealtimeTranscribeEvent = {
        type: 'error',
        sliceIndex: item.sliceIndex,
        data: undefined,
        isCapturing: this.audioStream.isRecording(),
        processTime: Date.now() - startTime,
        recordingTime: 0,
        memoryUsage: this.sliceManager.getMemoryUsage(),
        vadEvent: this.vadEvents.get(item.sliceIndex),
      }

      this.callbacks.onTranscribe?.(errorEvent)

      this.vadEvents.delete(item.sliceIndex)

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
   * Update VAD throttling options dynamically for low-end CPU optimization
   */
  updateVadThrottleOptions(options: {
    vadThrottleMs?: number
    vadSkipRatio?: number
  }): void {
    if (options.vadThrottleMs !== undefined) {
      this.options.vadThrottleMs = options.vadThrottleMs
    }
    if (options.vadSkipRatio !== undefined) {
      this.options.vadSkipRatio = options.vadSkipRatio
    }
    this.log(
      `VAD throttle options updated: throttleMs=${this.options.vadThrottleMs}, skipRatio=${this.options.vadSkipRatio}`,
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
      audioStats: {
        isRecording: this.audioStream.isRecording(),
        accumulatedSamples: this.accumulatedData.length,
      },
      vadStats: this.vadEnabled
        ? {
            enabled: true,
            contextAvailable: !!this.vadContext,
            lastSpeechDetectedTime: this.lastSpeechDetectedTime,
            isProcessing: this.isProcessingVAD,
            queueSize: this.vadProcessingQueue.length,
            skippedCount: this.skippedVadCount,
            throttleMs: this.options.vadThrottleMs,
            skipRatio: this.options.vadSkipRatio,
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

      // Process VAD for the slice if enabled (with throttling for low-end CPU)
      if (!this.isTranscribing && this.vadEnabled) {
        this.queueVADProcessing(result.slice)
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

    // Reset VAD throttling state
    this.isProcessingVAD = false
    this.lastVadProcessTime = 0
    this.vadProcessingQueue = []
    this.skippedVadCount = 0

    // Reset stats snapshot for clean start
    this.lastStatsSnapshot = null

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
