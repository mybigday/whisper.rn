/* eslint-disable class-methods-use-this */
import type {
  ParakeetTranscribeOptions,
  TranscribeOptions,
  TranscribeResult,
  VadOptions,
} from '../index'
import { SliceManager } from './SliceManager'
import { WavFileWriter, WavFileWriterFs } from '../utils/WavFileWriter'
import { StreamVad } from './StreamVad'
import type { StreamVadContextLike } from './StreamVad'
import { ContinuousSpeechSource } from './ContinuousSpeechSource'
import { LanguageLock } from './LanguageLock'
import {
  dropRepeatedPrefix,
  joinUnits,
  keepUnitsInRange,
  timedUnits,
} from './overlapAlignment'
import type { TimedUnit } from './overlapAlignment'
import { VAD_PRESETS } from './types'
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
  ParakeetContextLike,
  RealtimeSliceEndReason,
  RealtimeSegmenterLike,
  VadSpeechChunkInfo,
  VadSpeechEndInfo,
} from './types'

type TranscriptionItem = {
  sliceIndex: number
  audioData: Uint8Array
  /** Stream sample of audioData[0] */
  startSample: number
  isFinal?: boolean
  /** Final transcriptions: why the slice ended */
  endReason?: RealtimeSliceEndReason
  /** Overlap around in-speech splits: keep the words starting in [keepFrom, keepUntil) */
  keepFrom?: number
  keepUntil?: number
}

type TranscriptionRequest = {
  stop: () => Promise<void>
  promise: Promise<TranscribeResult>
}

const SILENCE_SEGMENT_REGEX = /\[(\s*\w+\s*)]/i

/** Slice length used when the options leave maxSpeechDurationS unbounded: Whisper's window */
const DEFAULT_MAX_SEGMENT_SEC = 30

/** Overlap word assignment: a word starting this much before the cut may still be kept */
const OVERLAP_SLACK_SEC = 0.1

/** Treat undefined / non-positive limits as "unbounded". */
const normalizeLimit = (value: number | undefined): number =>
  value && value > 0 ? Math.floor(value) : Infinity

/** Drop the oldest entries (Map insertion order) until the map fits the limit. */
const trimMapToLimit = (map: Map<number, unknown>, limit: number): void => {
  const keys = map.keys()
  let excess = map.size - limit
  while (excess > 0) {
    const { value: oldestKey } = keys.next()
    if (oldestKey === undefined) break
    map.delete(oldestKey)
    excess -= 1
  }
}

/**
 * Whether a transcript is a repetition loop ("A B C. A B C. A B C ..."): at
 * least 12 words (or CJK characters) of which most 3-grams are repeats.
 */
export const isRepetitive = (text: string): boolean => {
  const units = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]/.test(text)
    ? [...text.replace(/\s+/g, '')]
    : text.toLowerCase().split(/\s+/).filter(Boolean)
  if (units.length < 12) return false
  const grams = new Set<string>()
  for (let i = 0; i + 3 <= units.length; i += 1) grams.add(units.slice(i, i + 3).join(' '))
  return grams.size / (units.length - 2) < 0.5
}

/** The text up to its last sentence end (`.`, `!`, `?` and their CJK forms), '' if there is none */
export const untilLastSentenceEnd = (text: string): string => {
  // A CJK mark ends a sentence anywhere; a Latin one only before a space or the end
  const match = /^[\s\S]*(?:[\u3002\uff01\uff1f][\u300d\u300f\u201d)]*|[.!?]["'\u201d)\]]*(?=\s|$))/u.exec(
    text.trim(),
  )
  return match ? match[0] : ''
}

const isStreamVadContext = (value: unknown): value is StreamVadContextLike =>
  !!value &&
  typeof (value as StreamVadContextLike).createStream === 'function' &&
  typeof (value as StreamVadContextLike).hopSamples === 'number'

/**
 * RealtimeTranscriber transcribes a live audio stream in slices.
 *
 * - The audio goes through a segmenter: each speech segment becomes a slice.
 *   Given a speech probability source (a `WhisperVadContext`, or "always
 *   speech" without a VAD) the transcriber segments the probabilities itself
 *   (`StreamVad`): segments longer than `maxSpeechDurationS` are split, at an
 *   earlier pause when there is one. A segmenter VAD (`RealtimeSegmenterLike`)
 *   is used as it is.
 * - While a slice grows it gets draft transcriptions (`isFinal: false`); when
 *   it ends, one final transcription (`isFinal: true`, with `endReason`).
 * - Finals run before drafts, and a running draft is aborted for a final.
 * - Positions are counted in samples of the audio, not wall-clock time, so a
 *   recorded stream replays the same way.
 */
export class RealtimeTranscriber {
  private transcriptionContext:
    | { type: 'whisper'; context: WhisperContextLike }
    | { type: 'parakeet'; context: ParakeetContextLike }

  /** Audio -> speech segments with their audio */
  private segmentation: RealtimeSegmenterLike

  /** Bound on the length of a slice, in samples */
  private readonly maxSliceSamples: number

  /** Samples pushed to the segmenter (estimates positions a segmenter does not report) */
  private pushedSamples = 0

  private audioStream: AudioStreamInterface

  private fs?: WavFileWriterFs

  private sliceManager: SliceManager

  private callbacks: RealtimeTranscriberCallbacks = {}

  private options: {
    maxSlicesInMemory: number
    maxResultsInMemory: number
    transcribeOptions: TranscribeOptions | ParakeetTranscribeOptions
    draftTranscribeOptions: Partial<TranscribeOptions> | Partial<ParakeetTranscribeOptions>
    initialPrompt?: string
    promptPreviousSlices: boolean
    maxPromptSlices: number
    minAvgLogprob?: number
    audioOutputPath?: string
    audioStreamConfig?: AudioStreamConfig
    realtimeProcessingPauseMs: number
    initRealtimeAfterMs: number
    logger: (message: string) => void
  }

  private readonly sampleRate: number

  private isActive = false

  private isStopping = false

  private isTranscribing = false

  private vadEnabled = false

  private isSpeechActive = false

  private transcriptionQueue: TranscriptionItem[] = []

  private running: { item: TranscriptionItem; request: TranscriptionRequest } | null = null

  private wavFileWriter: WavFileWriter | null = null

  /** Stream sample of the current slice's first byte */
  private sliceStartSample = 0

  /**
   * Audio before this stream sample is in a finalized slice. Audio the
   * segmentation sends again after a split back is trimmed to it, so no audio
   * is transcribed in two finals.
   */
  private committedUntil = 0

  /** Stream sample of the current slice's audio end when its last draft was queued */
  private lastDraftEndSample = -Infinity

  /** Overlap (whisper, splitOverlapMs) */
  private readonly overlapSamples: number

  /** Keep-from bound of the current slice (it starts with overlap audio) */
  private sliceKeepFrom: number | undefined

  /** Audio before an in-speech split, to start the next slice with */
  private overlapCarry: { cut: number; data: Uint8Array } | null = null

  /** Units kept by the previous final, for the repeated-prefix check */
  private previousKeptUnits: TimedUnit[] = []

  private languageLock?: LanguageLock

  /** Slices that ended at a split inside speech (their text ends mid-sentence) */
  private inSpeechCuts = new Set<number>()

  /** Prompt chaining across an in-speech split (evaluation toggle) */
  private readonly promptAcrossSplit: 'full' | 'sentence' | 'drop'

  // Simplified VAD state management
  private lastSpeechDetectedTime = 0

  // Track last stats to emit only when changed
  private lastStatsSnapshot: any = null

  // Store transcription results by slice index. Bounded by maxResultsInMemory so
  // long-running sessions don't retain every slice's transcript and segments forever.
  private transcriptionResults: Map<
    number,
    { slice: AudioSliceNoData; transcribeEvent: RealtimeTranscribeEvent }
  > = new Map()

  // Store VAD events by slice index for inclusion in transcribe events. Normally deleted once
  // the slice's final transcription lands, but a slice that never reaches that point (filtered,
  // errored, empty) would leak its entry, so the map is also bounded by maxSlicesInMemory.
  private vadEvents: Map<number, RealtimeVadEvent> = new Map()

  // Track active async operations
  private activeTranscriptions: Set<{ promise: Promise<any> }> = new Set()

  constructor(
    dependencies: RealtimeTranscriberDependencies,
    options: RealtimeOptions = {},
    callbacks: RealtimeTranscriberCallbacks = {},
  ) {
    if (dependencies.parakeetContext) {
      this.transcriptionContext = {
        type: 'parakeet',
        context: dependencies.parakeetContext,
      }
    } else if (dependencies.whisperContext) {
      this.transcriptionContext = {
        type: 'whisper',
        context: dependencies.whisperContext,
      }
    } else {
      throw new Error(
        'RealtimeTranscriber requires either whisperContext or parakeetContext',
      )
    }
    const { vadContext } = dependencies
    this.audioStream = dependencies.audioStream
    this.fs = dependencies.fs
    this.callbacks = callbacks

    this.options = {
      maxSlicesInMemory: options.maxSlicesInMemory || 3,
      maxResultsInMemory: normalizeLimit(options.maxResultsInMemory),
      transcribeOptions: options.transcribeOptions || {},
      draftTranscribeOptions: options.draftTranscribeOptions || {},
      initialPrompt: options.initialPrompt,
      promptPreviousSlices: options.promptPreviousSlices ?? true,
      maxPromptSlices: normalizeLimit(options.maxPromptSlices),
      minAvgLogprob: options.minAvgLogprob,
      audioOutputPath: options.audioOutputPath,
      audioStreamConfig: options.audioStreamConfig,
      realtimeProcessingPauseMs: options.realtimeProcessingPauseMs ?? 200,
      initRealtimeAfterMs: options.initRealtimeAfterMs ?? 200,
      logger: options.logger || (() => { }),
    }
    const legacy = options as { audioSliceSec?: number; audioMinSec?: number }
    if (legacy.audioSliceSec !== undefined || legacy.audioMinSec !== undefined) {
      this.log(
        'audioSliceSec / audioMinSec are no longer used: slices follow the speech segments, ' +
          'bounded by vadOptions.maxSpeechDurationS',
      )
    }

    this.sampleRate = this.options.audioStreamConfig?.sampleRate || 16000
    this.vadEnabled = !!vadContext

    // Speech segmentation: the VAD's probabilities, or "always speech" cut by duration only
    const presetOptions = VAD_PRESETS[options.vadPreset ?? 'default']
    const vadOptions: VadOptions = vadContext
      ? { ...presetOptions, ...options.vadOptions }
      : {
        threshold: 0.5,
        minSpeechDurationMs: 0,
        minSilenceDurationMs: 100,
        speechPadMs: 0,
        maxSpeechDurationS:
          options.vadOptions?.maxSpeechDurationS ?? presetOptions.maxSpeechDurationS,
      }
    vadOptions.maxSpeechDurationS = this.boundedMaxSpeech(vadOptions.maxSpeechDurationS)
    this.maxSliceSamples = vadOptions.maxSpeechDurationS * this.sampleRate
    if (vadContext && !isStreamVadContext(vadContext)) {
      this.segmentation = vadContext
      this.log(
        'Using the VAD as a segmenter. Pass the probability source instead (e.g. the ' +
          'WhisperVadContext itself) to let the transcriber segment: stateful streaming and long ' +
          'speech split back at earlier pauses.',
      )
    } else {
      this.segmentation = new StreamVad(
        vadContext ?? new ContinuousSpeechSource(this.sampleRate),
        {
          vadOptions,
          segmenter: options.probabilitySegmenter,
          logger: (message) => this.log(message),
        },
      )
    }
    this.segmentation.onSpeechStart(this.handleSpeechDetected.bind(this))
    this.segmentation.onSpeechContinue(this.handleSpeechContinue.bind(this))
    this.segmentation.onSpeechEnd(this.handleSpeechEnded.bind(this))
    this.segmentation.onError(this.handleError.bind(this))

    const isWhisper = this.transcriptionContext.type === 'whisper'
    this.overlapSamples = isWhisper
      ? Math.max(0, Math.round(((options.splitOverlapMs ?? 0) * this.sampleRate) / 1000))
      : 0
    this.promptAcrossSplit =
      (options as { promptAcrossSplit?: 'full' | 'sentence' | 'drop' }).promptAcrossSplit ?? 'sentence'
    const { language } = this.options.transcribeOptions as TranscribeOptions
    if (isWhisper && options.languageLock && (!language || language === 'auto')) {
      this.languageLock = new LanguageLock(
        typeof options.languageLock === 'object' ? options.languageLock : {},
      )
    }

    this.sliceManager = new SliceManager(this.options.maxSlicesInMemory)

    // Set up audio stream callbacks
    this.audioStream.onData(this.handleAudioData.bind(this))
    this.audioStream.onError(this.handleError.bind(this))
    this.audioStream.onStatusChange(this.handleAudioStatusChange.bind(this))
    this.audioStream.onEnd?.(this.handleAudioEnd.bind(this))
  }

  /** A finite maximum segment duration: it bounds every slice, and so each draft's cost */
  private boundedMaxSpeech(value: number | undefined): number {
    if (value !== undefined && value > 0) return value
    this.log(`maxSpeechDurationS must be finite, using ${DEFAULT_MAX_SEGMENT_SEC} s`)
    return DEFAULT_MAX_SEGMENT_SEC
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
            sampleRate: this.sampleRate,
            channels: this.options.audioStreamConfig?.channels || 1,
            bitsPerSample: this.options.audioStreamConfig?.bitsPerSample || 16,
          },
        )
        await this.wavFileWriter.initialize()
      }

      // Start audio recording
      await this.audioStream.initialize({
        sampleRate: this.sampleRate,
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
   * Stop realtime transcription. Speech still open is closed and transcribed
   * (final, `endReason: 'end_of_stream'`) before this resolves; pending
   * drafts are dropped.
   */
  async stop(): Promise<void> {
    if (!this.isActive || this.isStopping) {
      return
    }

    try {
      this.isStopping = true

      // Stop audio recording first to stop new data coming in
      await this.audioStream.stop()

      // Close open speech: its final is queued like any other
      await this.segmentation.flush()
      this.finishSlice('end_of_stream')

      // Finals only from here
      this.transcriptionQueue = this.transcriptionQueue.filter((item) => item.isFinal)
      this.abortRunningDraft()
      await this.processTranscriptionQueue()
      await Promise.allSettled([...this.activeTranscriptions].map(t => t.promise))
      this.activeTranscriptions.clear()

      this.isActive = false
      await this.segmentation.reset()

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
      this.isActive = false
      this.handleError(`Stop error: ${error}`)
    } finally {
      this.isStopping = false
    }
  }

  /**
   * Handle incoming audio data
   */
  private handleAudioData(streamData: AudioStreamData): void {
    if (!this.isActive || this.isStopping) return

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
   * Process audio chunk through the speech segmentation
   */
  private async processAudioChunk(data: Uint8Array): Promise<void> {
    // Optional pre-VAD filter. Skipped audio is not part of the stream positions.
    if (this.callbacks.onBeginVad) {
      const duration = (data.length / 2) / (this.sampleRate / 1000) // ms
      const shouldContinue = await this.callbacks.onBeginVad({
        audioData: data,
        sliceIndex: -1, // No slice index yet for raw chunks
        duration,
      })
      if (!shouldContinue) return
    }

    this.pushedSamples += data.length / 2
    this.segmentation.processAudio(data)
  }

  // --- Segmentation handlers ---

  private handleSpeechDetected(
    confidence: number,
    data: Uint8Array,
    info?: VadSpeechChunkInfo,
  ): void {
    if (!this.isActive || this.isSpeechActive) return
    this.isSpeechActive = true
    this.log('Speech start')
    this.lastSpeechDetectedTime = Date.now()
    this.emitVadEvent('speech_start', confidence)

    // The previous segment's slice was finalized, so this audio starts a slice.
    // A segmenter without positions: assume its audio ends at what was pushed.
    const reported = info ? info.startSample : this.pushedSamples - data.length / 2
    const start = Math.max(reported, this.committedUntil)
    const carry = this.overlapCarry
    this.overlapCarry = null
    if (carry && carry.cut === start) {
      // Continues an in-speech split: start with the audio before the cut
      this.sliceStartSample = start - carry.data.length / 2
      this.sliceKeepFrom = start
      this.sliceManager.addAudioData(carry.data)
    } else {
      this.sliceStartSample = start
      this.sliceKeepFrom = undefined
    }
    this.appendAudio(data, info ? info.startSample : start)
    this.triggerDraft()
  }

  private handleSpeechContinue(
    confidence: number,
    data: Uint8Array,
    info?: VadSpeechChunkInfo,
  ): void {
    if (!this.isActive || !this.isSpeechActive) return
    this.emitVadEvent('speech_continue', confidence)
    if (info) {
      this.appendAudio(data, info.startSample)
    } else {
      this.sliceManager.addAudioData(data)
      // A segmenter that does not bound its segments: cut here, at the limit
      if (this.currentSliceAudio().length / 2 >= this.maxSliceSamples) {
        this.log('Slice reached maxSpeechDurationS, finalizing it')
        // The speech goes on in the next slice
        this.markInSpeechCut()
        this.finishSlice('max_duration')
      }
    }
    this.triggerDraft()
  }

  private handleSpeechEnded(confidence: number, info?: VadSpeechEndInfo): void {
    if (!this.isActive || !this.isSpeechActive) return
    this.isSpeechActive = false
    if (!info) {
      // No position: the slice ends with the audio received
      this.emitVadEvent('speech_end', confidence)
      this.finishSlice(undefined)
      return
    }

    // A split may end the segment before audio already received (that audio
    // comes again with the next segment). For an in-speech split with overlap,
    // the slice keeps overlapSamples past the cut and the next one starts
    // overlapSamples before it.
    const cut = info.endSample - this.sliceStartSample
    let keepUntil: number | undefined
    if (info.inSpeech && this.overlapSamples > 0) {
      const audio = this.currentSliceAudio()
      const available = audio.length / 2
      const end = Math.min(available, cut + this.overlapSamples)
      this.sliceManager.truncateCurrentSlice(Math.max(0, end) * 2)
      if (end > cut) keepUntil = info.endSample
      const from = Math.max(0, cut - this.overlapSamples)
      if (cut > from) {
        this.overlapCarry = { cut: info.endSample, data: audio.slice(from * 2, cut * 2) }
      }
    } else {
      this.sliceManager.truncateCurrentSlice(cut * 2)
    }
    this.emitVadEvent('speech_end', confidence)
    if (info.inSpeech) this.markInSpeechCut()
    this.finishSlice(info.reason, keepUntil)
  }

  /** The current slice ends at a split inside speech */
  private markInSpeechCut(): void {
    const index = this.sliceManager.getCurrentSliceInfo().currentSliceIndex
    this.inSpeechCuts.add(index)
    // Only the slice right before a prompt matters
    this.inSpeechCuts.forEach((old) => {
      if (old < index - 1) this.inSpeechCuts.delete(old)
    })
  }

  private currentSliceAudio(): Uint8Array {
    const index = this.sliceManager.getCurrentSliceInfo().currentSliceIndex
    return this.sliceManager.getAudioDataForTranscription(index) ?? new Uint8Array(0)
  }

  /**
   * Add audio to the current slice, without audio already in a finalized slice
   *
   * @param startSample Stream sample of `data[0]`
   */
  private appendAudio(data: Uint8Array, startSample: number): void {
    const committed = this.committedUntil - startSample
    const chunk = committed > 0 ? data.subarray(Math.min(data.length, committed * 2)) : data
    if (chunk.length > 0) this.sliceManager.addAudioData(chunk)
  }

  /**
   * Queue a draft of the current slice, throttled by audio time
   */
  private triggerDraft(): void {
    if (this.isStopping) return
    const { currentSliceIndex } = this.sliceManager.getCurrentSliceInfo()
    const audioData = this.sliceManager.getAudioDataForTranscription(currentSliceIndex)
    if (!audioData) return

    const samples = audioData.length / 2
    const endSample = this.sliceStartSample + samples
    const msToSamples = (ms: number) => (ms * this.sampleRate) / 1000
    if (samples < msToSamples(this.options.initRealtimeAfterMs)) return
    if (endSample - this.lastDraftEndSample < msToSamples(this.options.realtimeProcessingPauseMs)) {
      return
    }
    this.lastDraftEndSample = endSample

    // Only the newest draft is worth running: drafts waiting in the queue are stale
    this.transcriptionQueue = this.transcriptionQueue.filter((queued) => queued.isFinal)
    this.transcriptionQueue.push({
      sliceIndex: currentSliceIndex,
      audioData,
      startSample: this.sliceStartSample,
      isFinal: false,
    })
    this.processTranscriptionQueue().catch(e => this.handleError(e))
  }

  /**
   * Queue the final transcription of a slice: it goes before any draft, makes
   * drafts of this and earlier slices moot, and aborts a running draft.
   */
  private queueFinal(item: TranscriptionItem): void {
    this.transcriptionQueue = this.transcriptionQueue.filter(
      (queued) => queued.isFinal || queued.sliceIndex > item.sliceIndex,
    )
    const firstDraft = this.transcriptionQueue.findIndex((queued) => !queued.isFinal)
    if (firstDraft < 0) this.transcriptionQueue.push(item)
    else this.transcriptionQueue.splice(firstDraft, 0, item)
    this.abortRunningDraft()
    this.processTranscriptionQueue().catch((error) => {
      this.handleError(`Failed to process slice ${item.sliceIndex}: ${error}`)
    })
  }

  private abortRunningDraft(): void {
    if (this.running && !this.running.item.isFinal) {
      this.running.request.stop().catch((error) => {
        this.log(`Failed to abort a draft: ${error}`)
      })
    }
  }

  private emitVadEvent(type: RealtimeVadEvent['type'], confidence: number): void {
    // Without a VAD every audio is "speech": no VAD events to report
    if (!this.vadEnabled) return
    const sliceInfo = this.sliceManager.getCurrentSliceInfo()
    // Use the bytes written, not data.length: the slice buffer may be larger than its audio
    const sliceBytes = this.sliceManager.getSliceByIndex(sliceInfo.currentSliceIndex)?.sampleCount ?? 0
    const event: RealtimeVadEvent = {
      type,
      timestamp: Date.now(),
      sliceIndex: sliceInfo.currentSliceIndex,
      confidence,
      lastSpeechDetectedTime: this.lastSpeechDetectedTime,
      duration: sliceBytes / 2 / this.sampleRate, // seconds
    }
    this.vadEvents.set(sliceInfo.currentSliceIndex, event)
    trimMapToLimit(this.vadEvents, this.options.maxSlicesInMemory)
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
      promptParts.push(...this.getPreviousResultTexts(currentSliceIndex))
    }

    return promptParts.join(' ') || undefined
  }

  /**
   * Texts of the most recent non-empty results before the given slice, oldest first,
   * limited to maxPromptSlices so the prompt stays bounded regardless of session length.
   */
  private getPreviousResultTexts(currentSliceIndex: number): string[] {
    const previous: Array<[number, string]> = []
    this.transcriptionResults.forEach((result, sliceIndex) => {
      let text = result.transcribeEvent.data?.result
      if (text && sliceIndex === currentSliceIndex - 1 && this.inSpeechCuts.has(sliceIndex)) {
        // The slice right before was cut inside speech: its text ends mid-sentence
        // where this audio goes on, and Whisper tends to skip what it believes
        // the prompt already covers
        if (this.promptAcrossSplit === 'drop') text = ''
        else if (this.promptAcrossSplit === 'sentence') text = untilLastSentenceEnd(text)
      }
      // A repetition loop in the prompt makes Whisper repeat it in every later slice
      if (sliceIndex < currentSliceIndex && text && !isRepetitive(text)) {
        previous.push([sliceIndex, text])
      }
    })
    previous.sort(([a], [b]) => a - b)

    const { maxPromptSlices } = this.options
    if (previous.length > maxPromptSlices) {
      previous.splice(0, previous.length - maxPromptSlices)
    }
    return previous.map(([, text]) => text)
  }

  /** Request one transcription of an item; `autoLanguage` says whether Whisper detects the language */
  private requestTranscription(
    item: TranscriptionItem,
    audioBuffer: ArrayBuffer,
    durationMs: number,
  ): { request: TranscriptionRequest; autoLanguage: boolean } {
    const overrides = item.isFinal ? {} : this.options.draftTranscribeOptions
    if (this.transcriptionContext.type === 'parakeet') {
      const merged = { ...this.options.transcribeOptions, ...overrides }
      const parakeetOptions: ParakeetTranscribeOptions = {}
      if (merged.maxThreads !== undefined) parakeetOptions.maxThreads = merged.maxThreads
      // Parakeet only takes a fixed size; 'auto' is Whisper-only.
      if (typeof merged.audioCtx === 'number') parakeetOptions.audioCtx = merged.audioCtx
      return {
        request: this.transcriptionContext.context.transcribeData(audioBuffer, parakeetOptions),
        autoLanguage: false,
      }
    }

    // Prompt chaining, language lock and overlap tokens are Whisper-only.
    const options: TranscribeOptions & { onProgress?: undefined } = {
      ...(this.options.transcribeOptions as TranscribeOptions),
      ...(overrides as Partial<TranscribeOptions>),
      prompt: this.buildPrompt(item.sliceIndex),
      onProgress: undefined,
    }
    let autoLanguage = !options.language || options.language === 'auto'
    if (this.languageLock) {
      const locked = this.languageLock.languageFor(!!item.isFinal, durationMs)
      options.language = locked ?? 'auto'
      autoLanguage = !locked
    }
    if (item.isFinal && this.needsUnits(item)) {
      options.includeTokens = true
      options.tokenTimestamps = true
    }
    return {
      request: this.transcriptionContext.context.transcribeData(audioBuffer, options),
      autoLanguage,
    }
  }

  private needsUnits(item: TranscriptionItem): boolean {
    return this.overlapSamples > 0 && (item.keepFrom !== undefined || item.keepUntil !== undefined)
  }

  /** Text of a final whose slice has overlap audio: the words it keeps, without repeats */
  private applyOverlap(item: TranscriptionItem, result: TranscribeResult): void {
    if (!item.isFinal || this.overlapSamples === 0) return
    if (!this.needsUnits(item)) {
      this.previousKeptUnits = []
      return
    }
    const units = timedUnits(result.segments, item.startSample, this.sampleRate)
    if (!units) {
      this.log(`Slice ${item.sliceIndex}: no token times, overlap words kept as decoded`)
      this.previousKeptUnits = []
      return
    }
    let kept = keepUnitsInRange(
      units,
      item.keepFrom,
      item.keepUntil,
      OVERLAP_SLACK_SEC * this.sampleRate,
    )
    if (item.keepFrom !== undefined) kept = dropRepeatedPrefix(this.previousKeptUnits, kept)
    result.result = joinUnits(kept)
    this.previousKeptUnits = kept
  }

  /** Drop segments below minAvgLogprob (Whisper) */
  private applyLogprobFilter(result: TranscribeResult): void {
    const { minAvgLogprob } = this.options
    if (minAvgLogprob === undefined) return
    const kept = result.segments.filter(
      (segment) => segment.avgLogprob === undefined || segment.avgLogprob >= minAvgLogprob,
    )
    if (kept.length === result.segments.length) return
    result.segments = kept
    result.result = kept.map((segment) => segment.text).join('')
  }

  /**
   * Process a single transcription
   */
  private async processTranscription(item: TranscriptionItem): Promise<void> {
    if (!this.isActive || (this.isStopping && !item.isFinal)) {
      return
    }

    this.isTranscribing = true

    // Emit stats update for status change
    this.emitStatsUpdate('status_change')

    const startTime = Date.now()
    const durationMs = (item.audioData.length / 2) / (this.sampleRate / 1000)

    try {
      // Let the user filter this slice before paying for inference
      if (this.callbacks.onBeginTranscribe) {
        const shouldContinue = await this.callbacks.onBeginTranscribe({
          audioData: item.audioData,
          sliceIndex: item.sliceIndex,
          duration: durationMs,
          vadEvent: this.vadEvents.get(item.sliceIndex),
          isFinal: !!item.isFinal,
        })

        // Stopped while the callback was pending
        if (!this.isActive) return

        if (!shouldContinue) {
          this.log(`Transcription filtered by onBeginTranscribe for slice ${item.sliceIndex}`)
          return
        }
      }

      // The view may cover only part of its buffer (slices grow), so pass exactly its bytes
      const { buffer, byteOffset, byteLength } = item.audioData
      const audioBuffer = (
        byteOffset === 0 && byteLength === buffer.byteLength
          ? buffer
          : buffer.slice(byteOffset, byteOffset + byteLength)
      ) as ArrayBuffer
      const { request, autoLanguage } = this.requestTranscription(item, audioBuffer, durationMs)

      // Track active transcription
      this.activeTranscriptions.add(request)
      this.running = { item, request }

      let result: TranscribeResult
      try {
        result = await request.promise
      } finally {
        this.activeTranscriptions.delete(request)
        this.running = null
      }

      // Check if stopped during transcription
      if (!this.isActive) return

      if (result.isAborted) {
        // A draft aborted for a final (or a stop): nothing to report
        this.log(`Transcription of slice ${item.sliceIndex} aborted`)
        return
      }

      const endTime = Date.now()

      if (item.isFinal && autoLanguage && this.languageLock) {
        this.languageLock.observe(result.language, result.languageProb)
      }
      this.applyLogprobFilter(result)
      this.applyOverlap(item, result)
      // Normalize result and segments, remove "[ silence ]" or "[BLANK]"
      result.result = result.result.replace(SILENCE_SEGMENT_REGEX, '').trim()

      // The item carries its own audio: a backlog longer than maxSlicesInMemory
      // may have released the slice, but its transcription still counts
      const slice = this.sliceManager.getSliceByIndex(item.sliceIndex)

      // Create new transcription event
      const transcribeEvent: RealtimeTranscribeEvent = {
        type: 'transcribe',
        sliceIndex: item.sliceIndex,
        data: result,
        isCapturing: this.audioStream.isRecording(),
        processTime: endTime - startTime,
        recordingTime: durationMs,
        memoryUsage: this.sliceManager.getMemoryUsage(),
        vadEvent: this.vadEvents.get(item.sliceIndex),
        isFinal: !!item.isFinal,
        startSample: item.startSample,
        endSample: item.startSample + item.audioData.length / 2,
      }
      if (item.isFinal) transcribeEvent.endReason = item.endReason

      // if the current result is invalid, use the previous result
      const previousTranscribe = this.transcriptionResults.get(item.sliceIndex)
        ?.transcribeEvent
      if (previousTranscribe && result.result.trim() === '.') {
        transcribeEvent.data = previousTranscribe.data
      }

      // Save transcription results
      this.transcriptionResults.set(item.sliceIndex, {
        slice: {
          // Don't keep data in the slice
          index: item.sliceIndex,
          sampleCount: item.audioData.length,
          startTime: slice?.startTime ?? 0,
          endTime: slice?.endTime ?? 0,
          isProcessed: slice?.isProcessed ?? false,
          isReleased: slice?.isReleased ?? true,
        },
        transcribeEvent,
      })
      trimMapToLimit(this.transcriptionResults, this.options.maxResultsInMemory)

      // Emit transcribe event
      this.callbacks.onTranscribe?.(transcribeEvent)

      if (item.isFinal) {
        this.callbacks.onSliceTranscriptionStabilized?.(result.result?.trim() || '')
        this.vadEvents.delete(item.sliceIndex)
      }

      // Emit stats update for memory/slice changes
      this.emitStatsUpdate('memory_change')

      this.log(
        `Transcribed slice ${item.sliceIndex} (Final=${!!item.isFinal}): "${result.result}"`,
      )
    } catch (error) {
      this.handleError(`Transcription error: ${error}`)
    } finally {
      if (this.transcriptionQueue.length === 0) {
        this.isTranscribing = false
      }
    }
  }

  /**
   * Handle audio stream end: close open speech, its final is queued
   */
  private async handleAudioEnd(): Promise<void> {
    this.log('Audio stream ended')
    await this.segmentation.flush()
    this.finishSlice('end_of_stream')
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
   * Update segmentation options dynamically (maxSpeechDurationS stays finite)
   */
  updateVadOptions(options: Partial<VadOptions>): void {
    const next = { ...options }
    if ('maxSpeechDurationS' in next) {
      next.maxSpeechDurationS = this.boundedMaxSpeech(next.maxSpeechDurationS)
    }
    this.segmentation.updateOptions(next)
  }

  /** The language the language lock holds, if any */
  get lockedLanguage(): string | undefined {
    return this.languageLock?.locked
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
          contextAvailable: true,
          lastSpeechDetectedTime: this.lastSpeechDetectedTime,
        }
        : null,
      sliceStats: this.sliceManager.getCurrentSliceInfo(),
    }
  }

  /**
   * Get the retained transcription results, oldest first.
   *
   * When `maxResultsInMemory` is set this is only the most recent window; callers that need
   * every result should collect them from the `onTranscribe` callback as they arrive.
   */
  getTranscriptionResults(): Array<{
    slice: AudioSliceNoData
    transcribeEvent: RealtimeTranscribeEvent
  }> {
    return Array.from(this.transcriptionResults.values())
  }

  /**
   * Finalize the current slice now (`endReason: 'manual'`); speech that goes
   * on continues in the next slice.
   */
  async nextSlice(): Promise<void> {
    this.finishSlice('manual')
  }

  /**
   * Finalize the current slice and queue its final transcription
   *
   * @param keepUntil Stream sample of an in-speech cut the slice extends past (overlap)
   */
  private finishSlice(
    endReason: RealtimeSliceEndReason | undefined,
    keepUntil?: number,
  ): void {
    if (!this.isActive) {
      this.log('Cannot finish a slice - transcriber is not active')
      return
    }

    const result = this.sliceManager.forceNextSlice()
    if (!result.slice) return

    // Emit start event to indicate slice processing has started
    this.callbacks.onTranscribe?.({
      type: 'start',
      sliceIndex: -1, // Use -1 to indicate forced slice
      data: undefined,
      isCapturing: this.audioStream.isRecording(),
      processTime: 0,
      recordingTime: 0,
      memoryUsage: this.sliceManager.getMemoryUsage(),
    })

    const { slice } = result
    const startSample = this.sliceStartSample
    const sliceEnd = startSample + slice.data.length / 2
    this.log(`Slice ${slice.index} ready (${slice.data.length} bytes, ${endReason ?? 'ended'})`)

    // Positions move past it; with overlap, only up to the cut
    this.committedUntil = keepUntil ?? sliceEnd
    this.sliceStartSample = this.committedUntil
    const keepFrom = this.sliceKeepFrom
    this.sliceKeepFrom = undefined

    if (slice.data.length > 0) {
      const item: TranscriptionItem = {
        sliceIndex: slice.index,
        audioData: slice.data,
        startSample,
        isFinal: true,
        endReason,
      }
      if (keepFrom !== undefined) item.keepFrom = keepFrom
      if (keepUntil !== undefined) item.keepUntil = keepUntil
      this.queueFinal(item)
    }

    this.emitStatsUpdate('memory_change')
  }

  /**
   * Reset all components
   */
  reset(): void {
    this.sliceManager.reset()
    this.transcriptionQueue = []
    this.isTranscribing = false
    this.isSpeechActive = false

    // Reset VAD state
    this.lastSpeechDetectedTime = -1

    // Reset stats snapshot for clean start
    this.lastStatsSnapshot = null

    this.sliceStartSample = 0
    this.committedUntil = 0
    this.lastDraftEndSample = -Infinity
    this.sliceKeepFrom = undefined
    this.overlapCarry = null
    this.previousKeptUnits = []
    this.inSpeechCuts.clear()
    this.languageLock?.reset()

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

    this.segmentation.reset().catch(e => this.log(`Segmentation reset error: ${e}`))
  }

  /**
   * Release all resources. The VAD context stays the caller's; only the
   * stream this transcriber created on it is freed.
   */
  async release(): Promise<void> {
    if (this.isActive) {
      await this.stop()
    }

    await this.audioStream.release()
    await this.wavFileWriter?.finalize()
    await this.segmentation.release?.()
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
