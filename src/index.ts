import {
  NativeEventEmitter,
  DeviceEventEmitter,
  Platform,
  DeviceEventEmitterStatic,
  Image,
} from 'react-native'
import RNWhisper, {
  NativeWhisperContext,
  NativeWhisperVadContext,
} from './NativeRNWhisper'
import type {
  TranscribeOptions,
  TranscribeResult,
  CoreMLAsset,
  VadOptions,
  VadSegment,
} from './NativeRNWhisper'
import AudioSessionIos from './AudioSessionIos'
import type {
  AudioSessionCategoryIos,
  AudioSessionCategoryOptionIos,
  AudioSessionModeIos,
} from './AudioSessionIos'
import { version } from './version.json'

declare global {
  // eslint-disable-next-line no-var
  var whisperTranscribeData: (
    contextId: number,
    options: TranscribeOptions,
    data: ArrayBuffer,
  ) => Promise<TranscribeResult>
  // eslint-disable-next-line no-var
  var whisperVadDetectSpeech: (
    contextId: number,
    options: VadOptions,
    audioData: ArrayBuffer,
  ) => Promise<{ hasSpeech: boolean; segments: VadSegment[] }>
}

let jsiWhisperTranscribeData: (
  contextId: number,
  options: TranscribeOptions,
  data: ArrayBuffer,
) => Promise<TranscribeResult>
let jsiWhisperVadDetectSpeech: (
  contextId: number,
  options: VadOptions,
  audioData: ArrayBuffer,
) => Promise<{ hasSpeech: boolean; segments: VadSegment[] }>

let jsiInstalled = false

const installJSIBindingsIfNeeded = async () => {
  if (jsiInstalled) return
  jsiInstalled = true
  return RNWhisper.installJSIBindings()
    .then(() => {
      jsiWhisperTranscribeData = global.whisperTranscribeData
      delete (global as any).whisperTranscribeData
      jsiWhisperVadDetectSpeech = global.whisperVadDetectSpeech
      delete (global as any).whisperVadDetectSpeech
    })
    .catch((e) => {
      console.warn('Failed to install JSI bindings', e)
    })
}

let EventEmitter: NativeEventEmitter | DeviceEventEmitterStatic
if (Platform.OS === 'ios') {
  // @ts-ignore
  EventEmitter = new NativeEventEmitter(RNWhisper)
}
if (Platform.OS === 'android') {
  EventEmitter = DeviceEventEmitter
}

export type {
  TranscribeOptions,
  TranscribeResult,
  AudioSessionCategoryIos,
  AudioSessionCategoryOptionIos,
  AudioSessionModeIos,
  VadOptions,
  VadSegment,
}

const EVENT_ON_TRANSCRIBE_PROGRESS = '@RNWhisper_onTranscribeProgress'
const EVENT_ON_TRANSCRIBE_NEW_SEGMENTS = '@RNWhisper_onTranscribeNewSegments'
const EVENT_ON_NATIVE_LOG = '@RNWhisper_onNativeLog'

const EVENT_ON_REALTIME_TRANSCRIBE = '@RNWhisper_onRealtimeTranscribe'
const EVENT_ON_REALTIME_TRANSCRIBE_END = '@RNWhisper_onRealtimeTranscribeEnd'

const logListeners: Array<(level: string, text: string) => void> = []

// @ts-ignore
if (EventEmitter) {
  EventEmitter.addListener(
    EVENT_ON_NATIVE_LOG,
    (evt: { level: string; text: string }) => {
      logListeners.forEach((listener) => listener(evt.level, evt.text))
    },
  )
  // Trigger unset to use default log callback
  RNWhisper?.toggleNativeLog?.(false)?.catch?.(() => {})
}

export type TranscribeNewSegmentsResult = {
  nNew: number
  totalNNew: number
  result: string
  segments: TranscribeResult['segments']
}

export type TranscribeNewSegmentsNativeEvent = {
  contextId: number
  jobId: number
  result: TranscribeNewSegmentsResult
}

// Fn -> Boolean in TranscribeFileNativeOptions
export type TranscribeFileOptions = TranscribeOptions & {
  /**
   * Progress callback, the progress is between 0 and 100
   */
  onProgress?: (progress: number) => void
  /**
   * Callback when new segments are transcribed
   */
  onNewSegments?: (result: TranscribeNewSegmentsResult) => void
}

export type TranscribeProgressNativeEvent = {
  contextId: number
  jobId: number
  progress: number
}

export type AudioSessionSettingIos = {
  category: AudioSessionCategoryIos
  options?: AudioSessionCategoryOptionIos[]
  mode?: AudioSessionModeIos
  active?: boolean
}

// Codegen missing TSIntersectionType support so we dont put it into the native spec
export type TranscribeRealtimeOptions = TranscribeOptions & {
  /**
   * Realtime record max duration in seconds.
   * Due to the whisper.cpp hard constraint - processes the audio in chunks of 30 seconds,
   * the recommended value will be <= 30 seconds. (Default: 30)
   */
  realtimeAudioSec?: number
  /**
   * Optimize audio transcription performance by slicing audio samples when `realtimeAudioSec` > 30.
   * Set `realtimeAudioSliceSec` < 30 so performance improvements can be achieved in the Whisper hard constraint (processes the audio in chunks of 30 seconds).
   * (Default: Equal to `realtimeMaxAudioSec`)
   */
  realtimeAudioSliceSec?: number
  /**
   * Min duration of audio to start transcribe in seconds for each slice.
   * The minimum value is 0.5 ms and maximum value is realtimeAudioSliceSec (Default: 1)
   */
  realtimeAudioMinSec?: number
  /**
   * Output path for audio file. If not set, the audio file will not be saved
   * (Default: Undefined)
   */
  audioOutputPath?: string
  /**
   * Start transcribe on recording when the audio volume is greater than the threshold by using VAD (Voice Activity Detection).
   * The first VAD will be triggered after 2 second of recording.
   * (Default: false)
   */
  useVad?: boolean
  /**
   * The length of the collected audio is used for VAD, cannot be less than 2000ms. (ms) (Default: 2000)
   */
  vadMs?: number
  /**
   * VAD threshold. (Default: 0.6)
   */
  vadThold?: number
  /**
   * Frequency to apply High-pass filter in VAD. (Default: 100.0)
   */
  vadFreqThold?: number
  /**
   * iOS: Audio session settings when start transcribe
   * Keep empty to use current audio session state
   */
  audioSessionOnStartIos?: AudioSessionSettingIos
  /**
   * iOS: Audio session settings when stop transcribe
   * - Keep empty to use last audio session state
   * - Use `restore` to restore audio session state before start transcribe
   */
  audioSessionOnStopIos?: string | AudioSessionSettingIos
}

export type TranscribeRealtimeEvent = {
  contextId: number
  jobId: number
  /** Is capturing audio, when false, the event is the final result */
  isCapturing: boolean
  isStoppedByAction?: boolean
  code: number
  data?: TranscribeResult
  error?: string
  processTime: number
  recordingTime: number
  slices?: Array<{
    code: number
    error?: string
    data?: TranscribeResult
    processTime: number
    recordingTime: number
  }>
}

export type TranscribeRealtimeNativePayload = {
  /** Is capturing audio, when false, the event is the final result */
  isCapturing: boolean
  isStoppedByAction?: boolean
  code: number
  processTime: number
  recordingTime: number
  isUseSlices: boolean
  sliceIndex: number
  data?: TranscribeResult
  error?: string
}

export type TranscribeRealtimeNativeEvent = {
  contextId: number
  jobId: number
  payload: TranscribeRealtimeNativePayload
}

export type BenchResult = {
  config: string
  nThreads: number
  encodeMs: number
  decodeMs: number
  batchMs: number
  promptMs: number
}

const updateAudioSession = async (setting: AudioSessionSettingIos) => {
  await AudioSessionIos.setCategory(setting.category, setting.options || [])
  if (setting.mode) {
    await AudioSessionIos.setMode(setting.mode)
  }
  await AudioSessionIos.setActive(setting.active ?? true)
}

export class WhisperContext {
  ptr: number

  id: number

  gpu: boolean = false

  reasonNoGPU: string = ''

  constructor({
    contextPtr,
    contextId,
    gpu,
    reasonNoGPU,
  }: NativeWhisperContext) {
    this.ptr = contextPtr
    this.id = contextId
    this.gpu = gpu
    this.reasonNoGPU = reasonNoGPU
  }

  private transcribeWithNativeMethod(
    method: 'transcribeFile' | 'transcribeData',
    data: string,
    options: TranscribeFileOptions = {},
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  } {
    const jobId: number = Math.floor(Math.random() * 10000)

    const { onProgress, onNewSegments, ...rest } = options

    let progressListener: any
    let lastProgress: number = 0
    if (onProgress) {
      progressListener = EventEmitter.addListener(
        EVENT_ON_TRANSCRIBE_PROGRESS,
        (evt: TranscribeProgressNativeEvent) => {
          const { contextId, progress } = evt
          if (contextId !== this.id || evt.jobId !== jobId) return
          lastProgress = progress > 100 ? 100 : progress
          onProgress(lastProgress)
        },
      )
    }
    const removeProgressListener = () => {
      if (progressListener) {
        progressListener.remove()
        progressListener = null
      }
    }

    let newSegmentsListener: any
    if (onNewSegments) {
      newSegmentsListener = EventEmitter.addListener(
        EVENT_ON_TRANSCRIBE_NEW_SEGMENTS,
        (evt: TranscribeNewSegmentsNativeEvent) => {
          const { contextId, result } = evt
          if (contextId !== this.id || evt.jobId !== jobId) return
          onNewSegments(result)
        },
      )
    }
    const removeNewSegmenetsListener = () => {
      if (newSegmentsListener) {
        newSegmentsListener.remove()
        newSegmentsListener = null
      }
    }

    return {
      stop: async () => {
        await RNWhisper.abortTranscribe(this.id, jobId)
        removeProgressListener()
        removeNewSegmenetsListener()
      },
      promise: RNWhisper[method](this.id, jobId, data, {
        ...rest,
        onProgress: !!onProgress,
        onNewSegments: !!onNewSegments,
      })
        .then((result) => {
          removeProgressListener()
          removeNewSegmenetsListener()
          if (!result.isAborted && lastProgress !== 100) {
            // Handle the case that the last progress event is not triggered
            onProgress?.(100)
          }
          return result
        })
        .catch((e) => {
          removeProgressListener()
          removeNewSegmenetsListener()
          throw e
        }),
    }
  }

  /**
   * Transcribe audio file (path or base64 encoded wav file)
   * base64: need add `data:audio/wav;base64,` prefix
   */
  transcribe(
    filePathOrBase64: string | number,
    options: TranscribeFileOptions = {},
  ): {
    /** Stop the transcribe */
    stop: () => Promise<void>
    /** Transcribe result promise */
    promise: Promise<TranscribeResult>
  } {
    let path = ''
    if (typeof filePathOrBase64 === 'number') {
      try {
        const source = Image.resolveAssetSource(filePathOrBase64)
        if (source) path = source.uri
      } catch (e) {
        throw new Error(`Invalid asset: ${filePathOrBase64}`)
      }
    } else {
      if (filePathOrBase64.startsWith('http'))
        throw new Error(
          'Transcribe remote file is not supported, please download it first',
        )
      path = filePathOrBase64
    }
    if (path.startsWith('file://')) path = path.slice(7)
    return this.transcribeWithNativeMethod('transcribeFile', path, options)
  }

  /**
   * Transcribe audio data (base64 encoded float32 PCM data or ArrayBuffer)
   */
  transcribeData(
    data: string | ArrayBuffer,
    options: TranscribeFileOptions = {},
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  } {
    if (data instanceof ArrayBuffer) {
      // Use JSI function for ArrayBuffer
      if (!jsiWhisperTranscribeData) {
        throw new Error('JSI binding `whisperTranscribeData` not installed')
      }
      return this.transcribeDataArrayBuffer(data, options)
    }
    return this.transcribeWithNativeMethod('transcribeData', data, options)
  }

  /**
   * Transcribe audio data from ArrayBuffer (16-bit PCM, mono, 16kHz)
   */
  private transcribeDataArrayBuffer(
    data: ArrayBuffer,
    options: TranscribeFileOptions = {},
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  } {
    const { onProgress, onNewSegments, ...rest } = options

    // Generate a unique jobId for this transcription
    const jobId = Math.floor(Math.random() * 10000)

    const jsiOptions = {
      ...rest,
      onProgress: onProgress || undefined,
      onNewSegments: onNewSegments || undefined,
      jobId, // Pass jobId to native implementation
    }

    let isAborted = false
    const promise = jsiWhisperTranscribeData(this.id, jsiOptions, data)
      .then((result: any) => {
        if (isAborted) {
          return { ...result, isAborted: true }
        }
        return result
      })
      .catch((error: any) => {
        if (isAborted) {
          return { isAborted: true, error: 'Transcription aborted' }
        }
        throw error
      })

    return {
      stop: async () => {
        isAborted = true
        try {
          // Use the existing native abort method
          await RNWhisper.abortTranscribe(this.id, jobId)
        } catch (error) {
          // Ignore errors if context is already released or job doesn't exist
        }
      },
      promise,
    }
  }

  /** Transcribe the microphone audio stream, the microphone user permission is required */
  async transcribeRealtime(options: TranscribeRealtimeOptions = {}): Promise<{
    /** Stop the realtime transcribe */
    stop: () => Promise<void>
    /** Subscribe to realtime transcribe events */
    subscribe: (callback: (event: TranscribeRealtimeEvent) => void) => void
  }> {
    console.warn(
      '`transcribeRealtime` is deprecated, use `RealtimeTranscriber` instead',
    )

    let lastTranscribePayload: TranscribeRealtimeNativePayload

    const slices: TranscribeRealtimeNativePayload[] = []
    let sliceIndex: number = 0
    let tOffset: number = 0

    const putSlice = (payload: TranscribeRealtimeNativePayload) => {
      if (!payload.isUseSlices || !payload.data) return
      if (sliceIndex !== payload.sliceIndex) {
        const { segments = [] } = slices[sliceIndex]?.data || {}
        tOffset = segments[segments.length - 1]?.t1 || 0
      }
      ;({ sliceIndex } = payload)
      slices[sliceIndex] = {
        ...payload,
        data: {
          ...payload.data,
          segments:
            payload.data.segments.map((segment) => ({
              ...segment,
              t0: segment.t0 + tOffset,
              t1: segment.t1 + tOffset,
            })) || [],
        },
      }
    }

    const mergeSlicesIfNeeded = (
      payload: TranscribeRealtimeNativePayload,
    ): TranscribeRealtimeNativePayload => {
      if (!payload.isUseSlices) return payload

      const mergedPayload: any = {}
      slices.forEach((slice) => {
        mergedPayload.data = {
          result:
            (mergedPayload.data?.result || '') + (slice.data?.result || ''),
          segments: [
            ...(mergedPayload?.data?.segments || []),
            ...(slice.data?.segments || []),
          ],
        }
        mergedPayload.processTime = slice.processTime
        mergedPayload.recordingTime =
          (mergedPayload?.recordingTime || 0) + slice.recordingTime
      })
      return { ...payload, ...mergedPayload, slices }
    }

    let prevAudioSession: AudioSessionSettingIos | undefined
    if (Platform.OS === 'ios' && options?.audioSessionOnStartIos) {
      // iOS: Remember current audio session state
      if (options?.audioSessionOnStopIos === 'restore') {
        const categoryResult = await AudioSessionIos.getCurrentCategory()
        const mode = await AudioSessionIos.getCurrentMode()

        prevAudioSession = {
          ...categoryResult,
          mode,
          active: false, // TODO: Need to check isOtherAudioPlaying to set active
        }
      }

      // iOS: Update audio session state
      await updateAudioSession(options?.audioSessionOnStartIos)
    }
    if (
      Platform.OS === 'ios' &&
      typeof options?.audioSessionOnStopIos === 'object'
    ) {
      prevAudioSession = options?.audioSessionOnStopIos
    }

    const jobId: number = Math.floor(Math.random() * 10000)
    try {
      await RNWhisper.startRealtimeTranscribe(this.id, jobId, options)
    } catch (e) {
      if (prevAudioSession) await updateAudioSession(prevAudioSession)
      throw e
    }

    return {
      stop: async () => {
        await RNWhisper.abortTranscribe(this.id, jobId)
        if (prevAudioSession) await updateAudioSession(prevAudioSession)
      },
      subscribe: (callback: (event: TranscribeRealtimeEvent) => void) => {
        let transcribeListener: any = EventEmitter.addListener(
          EVENT_ON_REALTIME_TRANSCRIBE,
          (evt: TranscribeRealtimeNativeEvent) => {
            const { contextId, payload } = evt
            if (contextId !== this.id || evt.jobId !== jobId) return
            lastTranscribePayload = payload
            putSlice(payload)
            callback({
              contextId,
              jobId: evt.jobId,
              ...mergeSlicesIfNeeded(payload),
            })
          },
        )
        let endListener: any = EventEmitter.addListener(
          EVENT_ON_REALTIME_TRANSCRIBE_END,
          (evt: TranscribeRealtimeNativeEvent) => {
            const { contextId, payload } = evt
            if (contextId !== this.id || evt.jobId !== jobId) return
            const lastPayload = {
              ...lastTranscribePayload,
              ...payload,
            }
            putSlice(lastPayload)
            callback({
              contextId,
              jobId: evt.jobId,
              ...mergeSlicesIfNeeded(lastPayload),
              isCapturing: false,
            })
            if (transcribeListener) {
              transcribeListener.remove()
              transcribeListener = null
            }
            if (endListener) {
              endListener.remove()
              endListener = null
            }
          },
        )
      },
    }
  }

  async bench(maxThreads: number): Promise<BenchResult> {
    const result = await RNWhisper.bench(this.id, maxThreads)
    const [config, nThreads, encodeMs, decodeMs, batchMs, promptMs] =
      JSON.parse(result)
    return {
      config,
      nThreads,
      encodeMs,
      decodeMs,
      batchMs,
      promptMs,
    } as BenchResult
  }

  async release(): Promise<void> {
    return RNWhisper.releaseContext(this.id)
  }
}

export type ContextOptions = {
  filePath: string | number
  /**
   * CoreML model assets, if you're using `require` on filePath,
   * use this option is required if you want to enable Core ML,
   * you will need bundle weights/weight.bin, model.mil, coremldata.bin into app by `require`
   */
  coreMLModelAsset?: {
    filename: string
    assets: string[] | number[]
  }
  /** Is the file path a bundle asset for pure string filePath */
  isBundleAsset?: boolean
  /** Prefer to use Core ML model if exists. If set to false, even if the Core ML model exists, it will not be used. */
  useCoreMLIos?: boolean
  /** Use GPU if available. Currently iOS only, if it's enabled, Core ML option will be ignored. */
  useGpu?: boolean
  /** Use Flash Attention, only recommended if GPU available */
  useFlashAttn?: boolean
}

const coreMLModelAssetPaths = [
  'analytics/coremldata.bin',
  'weights/weight.bin',
  'model.mil',
  'coremldata.bin',
]

export async function initWhisper({
  filePath,
  coreMLModelAsset,
  isBundleAsset,
  useGpu = true,
  useCoreMLIos = true,
  useFlashAttn = false,
}: ContextOptions): Promise<WhisperContext> {
  await installJSIBindingsIfNeeded()

  let path = ''
  let coreMLAssets: CoreMLAsset[] | undefined
  if (coreMLModelAsset) {
    const { filename, assets } = coreMLModelAsset
    if (filename && assets) {
      coreMLAssets = assets
        ?.map((asset) => {
          if (typeof asset === 'number') {
            const { uri } = Image.resolveAssetSource(asset)
            const filepath = coreMLModelAssetPaths.find((p) => uri.includes(p))
            if (filepath) {
              return {
                uri,
                filepath: `${filename}/${filepath}`,
              }
            }
          } else if (typeof asset === 'string') {
            return {
              uri: asset,
              filepath: `${filename}/${asset}`,
            }
          }
          return undefined
        })
        .filter((asset): asset is CoreMLAsset => asset !== undefined)
    }
  }
  if (typeof filePath === 'number') {
    try {
      const source = Image.resolveAssetSource(filePath)
      if (source) {
        path = source.uri
      }
    } catch (e) {
      throw new Error(`Invalid asset: ${filePath}`)
    }
  } else {
    if (!isBundleAsset && filePath.startsWith('http'))
      throw new Error(
        'Transcribe remote file is not supported, please download it first',
      )
    path = filePath
  }
  if (path.startsWith('file://')) path = path.slice(7)
  const { contextPtr, contextId, gpu, reasonNoGPU } =
    await RNWhisper.initContext({
      filePath: path,
      isBundleAsset: !!isBundleAsset,
      useFlashAttn,
      useGpu,
      useCoreMLIos,
      // Only development mode need download Core ML model assets (from packager server)
      downloadCoreMLAssets: __DEV__ && !!coreMLAssets,
      coreMLAssets,
    })
  return new WhisperContext({ contextPtr, contextId, gpu, reasonNoGPU })
}

export async function releaseAllWhisper(): Promise<void> {
  await installJSIBindingsIfNeeded()

  return RNWhisper.releaseAllContexts()
}

/** Current version of whisper.cpp */
export const libVersion: string = version

const { useCoreML, coreMLAllowFallback } = RNWhisper.getConstants?.() || {}

/** Is use CoreML models on iOS */
export const isUseCoreML: boolean = !!useCoreML

/** Is allow fallback to CPU if load CoreML model failed */
export const isCoreMLAllowFallback: boolean = !!coreMLAllowFallback

export { AudioSessionIos }

//
// VAD (Voice Activity Detection) Context
//

export type VadContextOptions = {
  filePath: string | number
  /** Is the file path a bundle asset for pure string filePath */
  isBundleAsset?: boolean
  /** Use GPU if available. Currently iOS only */
  useGpu?: boolean
  /** Number of threads to use during computation (Default: 2 for 4-core devices, 4 for more cores) */
  nThreads?: number
}

export class WhisperVadContext {
  id: number

  gpu: boolean = false

  reasonNoGPU: string = ''

  constructor({ contextId, gpu, reasonNoGPU }: NativeWhisperVadContext) {
    this.id = contextId
    this.gpu = gpu
    this.reasonNoGPU = reasonNoGPU
  }

  /**
   * Detect speech segments in audio file (path or base64 encoded wav file)
   * base64: need add `data:audio/wav;base64,` prefix
   */
  async detectSpeech(
    filePathOrBase64: string | number,
    options: VadOptions = {},
  ): Promise<VadSegment[]> {
    let path = ''
    if (typeof filePathOrBase64 === 'number') {
      try {
        const source = Image.resolveAssetSource(filePathOrBase64)
        if (source) path = source.uri
      } catch (e) {
        throw new Error(`Invalid asset: ${filePathOrBase64}`)
      }
    } else {
      if (filePathOrBase64.startsWith('http'))
        throw new Error(
          'VAD remote file is not supported, please download it first',
        )
      path = filePathOrBase64
    }
    if (path.startsWith('file://')) path = path.slice(7)

    // Check if this is base64 encoded audio data
    if (path.startsWith('data:audio/')) {
      // This is base64 encoded audio data, use the raw data method
      return RNWhisper.vadDetectSpeech(this.id, path, options)
    } else {
      // This is a file path, use the file method
      return RNWhisper.vadDetectSpeechFile(this.id, path, options)
    }
  }

  /**
   * Detect speech segments in raw audio data (base64 encoded float32 PCM data or ArrayBuffer)
   */
  async detectSpeechData(
    audioData: string | ArrayBuffer,
    options: VadOptions = {},
  ): Promise<VadSegment[]> {
    if (audioData instanceof ArrayBuffer) {
      // Use JSI function for ArrayBuffer
      if (!jsiWhisperVadDetectSpeech) {
        throw new Error('JSI binding `whisperVadDetectSpeech` not installed')
      }
      const result = await jsiWhisperVadDetectSpeech(
        this.id,
        options,
        audioData,
      )
      return result.segments || []
    }
    return RNWhisper.vadDetectSpeech(this.id, audioData, options)
  }

  async release(): Promise<void> {
    return RNWhisper.releaseVadContext(this.id)
  }
}

/**
 * Initialize a VAD context for voice activity detection
 * @param options VAD context options
 * @returns Promise resolving to WhisperVadContext instance
 */
export async function initWhisperVad({
  filePath,
  isBundleAsset,
  useGpu = true,
  nThreads,
}: VadContextOptions): Promise<WhisperVadContext> {
  await installJSIBindingsIfNeeded()

  let path = ''
  if (typeof filePath === 'number') {
    try {
      const source = Image.resolveAssetSource(filePath)
      if (source) {
        path = source.uri
      }
    } catch (e) {
      throw new Error(`Invalid asset: ${filePath}`)
    }
  } else {
    if (!isBundleAsset && filePath.startsWith('http'))
      throw new Error(
        'VAD remote file is not supported, please download it first',
      )
    path = filePath
  }
  if (path.startsWith('file://')) path = path.slice(7)
  const { contextId, gpu, reasonNoGPU } = await RNWhisper.initVadContext({
    filePath: path,
    isBundleAsset: !!isBundleAsset,
    useGpu,
    nThreads,
  })
  return new WhisperVadContext({ contextId, gpu, reasonNoGPU })
}

/**
 * Release all VAD contexts and free their memory
 * @returns Promise resolving when all contexts are released
 */
export async function releaseAllWhisperVad(): Promise<void> {
  await installJSIBindingsIfNeeded()

  return RNWhisper.releaseAllVadContexts()
}

let logInitialized = false

export async function toggleNativeLog(enabled: boolean): Promise<void> {
  if (!enabled && !logInitialized) return // If first call is false, skip

  logInitialized = true
  await installJSIBindingsIfNeeded()

  return RNWhisper.toggleNativeLog(enabled)
}

export function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void } {
  logListeners.push(listener)
  return {
    remove: () => {
      logListeners.splice(logListeners.indexOf(listener), 1)
    },
  }
}
