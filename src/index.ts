import { Image, NativeModules } from 'react-native'
import { Buffer } from 'safe-buffer'
import RNWhisper from './NativeRNWhisper'
import './jsi'
import type {
  CoreMLAsset,
  NativeWhisperContext,
  NativeWhisperVadContext,
  NativeContextOptions,
  NativeVadContextOptions,
  TranscribeOptions,
  TranscribeResult,
  VadOptions,
  VadSegment,
} from './NativeRNWhisper'
import { version } from './version.json'

type NativeConstants = {
  useCoreML?: boolean
  coreMLAllowFallback?: boolean
}

type CoreMLModelAssetOptions = {
  filename: string
  assets: string[] | number[]
}

const nativeConstants: NativeConstants =
  ((RNWhisper as any)?.getConstants?.() as NativeConstants | undefined) ??
  (NativeModules.RNWhisper?.getConstants?.() as NativeConstants | undefined) ??
  {}

const jsiBindingKeys = [
  'whisperGetConstants',
  'whisperInitContext',
  'whisperReleaseContext',
  'whisperReleaseAllContexts',
  'whisperTranscribeFile',
  'whisperTranscribeData',
  'whisperAbortTranscribe',
  'whisperBench',
  'whisperInitVadContext',
  'whisperReleaseVadContext',
  'whisperReleaseAllVadContexts',
  'whisperVadDetectSpeech',
  'whisperVadDetectSpeechFile',
  'whisperToggleNativeLog',
] as const

type JsiBindingKey = (typeof jsiBindingKeys)[number]
type JsiBindings = { [K in JsiBindingKey]: NonNullable<(typeof globalThis)[K]> }

let jsiBindings: JsiBindings | null = null
let isJsiInstalled = false

const bindJsiFromGlobal = () => {
  const bindings: Partial<JsiBindings> = {}
  const missing: string[] = []

  jsiBindingKeys.forEach((key) => {
    const value = global[key]
    if (typeof value === 'function') {
      ;(bindings as Record<string, unknown>)[key] =
        value as JsiBindings[typeof key]
      delete (globalThis as any)[key]
    } else {
      missing.push(key)
    }
  })

  if (missing.length > 0) {
    throw new Error(`[RNWhisper] Missing JSI bindings: ${missing.join(', ')}`)
  }

  jsiBindings = bindings as JsiBindings
}

const getJsi = (): JsiBindings => {
  if (!jsiBindings) {
    throw new Error('JSI bindings not installed')
  }
  return jsiBindings
}

export const installJsi = async () => {
  if (isJsiInstalled) return

  if (typeof global.whisperInitContext !== 'function') {
    const installed = await RNWhisper.install()
    if (!installed && typeof global.whisperInitContext !== 'function') {
      throw new Error('JSI bindings not installed')
    }
  }

  bindJsiFromGlobal()
  isJsiInstalled = true
}

const toArrayBuffer = (view: Uint8Array): ArrayBuffer =>
  Uint8Array.from(view).buffer

const decodeBase64ToArrayBuffer = (data: string): ArrayBuffer =>
  toArrayBuffer(Buffer.from(data, 'base64') as unknown as Uint8Array)

const stripFileScheme = (path: string): string =>
  path.startsWith('file://') ? path.slice(7) : path

let contextIdCounter = 1
const contextIdRandom = () =>
  process.env.NODE_ENV === 'test'
    ? 0
    : Math.floor(Math.random() * 0x7fffffff)

const createContextId = (): number => {
  const contextId = contextIdCounter + contextIdRandom()
  contextIdCounter += 1
  return contextId
}

const coreMLModelAssetPaths = [
  'analytics/coremldata.bin',
  'weights/weight.bin',
  'model.mil',
  'coremldata.bin',
]

const resolvePathFromAsset = (asset: number): string => {
  try {
    const source = Image.resolveAssetSource(asset)
    if (source?.uri) {
      return source.uri
    }
  } catch (error) {
    throw new Error(`Invalid asset: ${asset}`)
  }
  throw new Error(`Invalid asset: ${asset}`)
}

const resolveLocalInputPath = (
  input: string | number,
  remoteError: string,
): string => {
  if (typeof input === 'number') {
    return resolvePathFromAsset(input)
  }

  if (input.startsWith('http://') || input.startsWith('https://')) {
    throw new Error(remoteError)
  }

  return stripFileScheme(input)
}

const createCoreMLAssets = (
  coreMLModelAsset?: CoreMLModelAssetOptions,
): CoreMLAsset[] | undefined => {
  if (!coreMLModelAsset?.filename || !coreMLModelAsset.assets) {
    return undefined
  }

  return coreMLModelAsset.assets
    .map((asset) => {
      if (typeof asset === 'number') {
        const { uri } = Image.resolveAssetSource(asset)
        const filepath = coreMLModelAssetPaths.find((path) =>
          uri.includes(path),
        )
        if (!filepath) return undefined
        return {
          uri,
          filepath: `${coreMLModelAsset.filename}/${filepath}`,
        }
      }

      return {
        uri: asset,
        filepath: `${coreMLModelAsset.filename}/${asset}`,
      }
    })
    .filter((asset): asset is CoreMLAsset => asset !== undefined)
}

const normalizeBenchResult = (result: string) => {
  const [config, nThreads, encodeMs, decodeMs, batchMs, promptMs] =
    JSON.parse(result)
  return {
    config,
    nThreads,
    encodeMs,
    decodeMs,
    batchMs,
    promptMs,
  }
}

const logListeners: Array<(level: string, text: string) => void> = []
const emitNativeLog = (level: string, text: string) => {
  logListeners.forEach((listener) => listener(level, text))
}

export type {
  TranscribeOptions,
  TranscribeResult,
  VadOptions,
  VadSegment,
}

export type TranscribeNewSegmentsResult = {
  nNew: number
  totalNNew: number
  result: string
  segments: TranscribeResult['segments']
}

export interface TranscribeFileOptions extends TranscribeOptions {
  /** Progress callback, the progress is between 0 and 100 */
  onProgress?: (progress: number) => void
  /** Callback when new segments are transcribed */
  onNewSegments?: (result: TranscribeNewSegmentsResult) => void
}

export type BenchResult = {
  config: string
  nThreads: number
  encodeMs: number
  decodeMs: number
  batchMs: number
  promptMs: number
}

export class WhisperContext {
  ptr: number

  id: number

  gpu = false

  reasonNoGPU = ''

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

  private runTranscription(
    run: (jobId: number) => Promise<TranscribeResult>,
  ): { stop: () => Promise<void>; promise: Promise<TranscribeResult> } {
    const { whisperAbortTranscribe } = getJsi()
    const jobId = Math.floor(Math.random() * 10000)

    return {
      stop: async () => {
        await whisperAbortTranscribe(this.id, jobId)
      },
      promise: run(jobId),
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
    const { whisperTranscribeFile } = getJsi()
    const { onProgress, ...rest } = options
    let lastProgress = 0
    const progressCallback = onProgress
      ? (progress: number) => {
          lastProgress = progress
          onProgress(progress)
        }
      : undefined

    let path = ''
    if (typeof filePathOrBase64 === 'number') {
      path = resolvePathFromAsset(filePathOrBase64)
    } else if (filePathOrBase64.startsWith('data:audio/wav;base64,')) {
      path = filePathOrBase64
    } else {
      path = resolveLocalInputPath(
        filePathOrBase64,
        'Transcribe remote file is not supported, please download it first',
      )
    }

    const task = this.runTranscription((jobId) =>
      whisperTranscribeFile(this.id, path, {
        ...rest,
        onProgress: progressCallback,
        jobId,
      }),
    )

    return {
      stop: task.stop,
      promise: task.promise.then((result) => {
        if (onProgress && !result.isAborted && lastProgress !== 100) {
          onProgress(100)
        }
        return result
      }),
    }
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
    const { whisperTranscribeData } = getJsi()
    const { onProgress, ...rest } = options
    let lastProgress = 0
    const progressCallback = onProgress
      ? (progress: number) => {
          lastProgress = progress
          onProgress(progress)
        }
      : undefined
    const audioData =
      data instanceof ArrayBuffer ? data : decodeBase64ToArrayBuffer(data)

    const task = this.runTranscription(
      (jobId) =>
        whisperTranscribeData(
          this.id,
          { ...rest, onProgress: progressCallback, jobId },
          audioData,
        ),
    )

    return {
      stop: task.stop,
      promise: task.promise.then((result) => {
        if (onProgress && !result.isAborted && lastProgress !== 100) {
          onProgress(100)
        }
        return result
      }),
    }
  }

  async bench(maxThreads: number): Promise<BenchResult> {
    const { whisperBench } = getJsi()
    const result = await whisperBench(this.id, maxThreads)
    return normalizeBenchResult(result)
  }

  async release(): Promise<void> {
    const { whisperReleaseContext } = getJsi()
    return whisperReleaseContext(this.id)
  }
}

export type ContextOptions = {
  filePath: string | number
  /**
   * CoreML model assets, if you're using `require` on filePath,
   * use this option is required if you want to enable Core ML,
   * you will need bundle weights/weight.bin, model.mil, coremldata.bin into app by `require`
   */
  coreMLModelAsset?: CoreMLModelAssetOptions
  /** Is the file path a bundle asset for pure string filePath */
  isBundleAsset?: boolean
  /** Prefer to use Core ML model if exists. If set to false, even if the Core ML model exists, it will not be used. */
  useCoreMLIos?: boolean
  /** Use GPU if available. Currently iOS only, if it's enabled, Core ML option will be ignored. */
  useGpu?: boolean
  /** Use Flash Attention, only recommended if GPU available */
  useFlashAttn?: boolean
}

/**
 * Initialize a whisper context with a GGML model file
 * @param options Whisper context options
 * @returns Promise resolving to WhisperContext instance
 */
export async function initWhisper({
  filePath,
  coreMLModelAsset,
  isBundleAsset,
  useGpu = true,
  useCoreMLIos = true,
  useFlashAttn = false,
}: ContextOptions): Promise<WhisperContext> {
  await installJsi()
  const { whisperInitContext } = getJsi()

  const coreMLAssets = createCoreMLAssets(coreMLModelAsset)
  const path =
    typeof filePath === 'number'
      ? resolvePathFromAsset(filePath)
      : resolveLocalInputPath(
          filePath,
          'Transcribe remote file is not supported, please download it first',
        )

  const contextId = createContextId()
  const context = await whisperInitContext(contextId, {
    filePath: path,
    isBundleAsset: !!isBundleAsset,
    useFlashAttn,
    useGpu,
    useCoreMLIos,
    downloadCoreMLAssets: __DEV__ && !!coreMLAssets,
    coreMLAssets,
  } satisfies NativeContextOptions)

  return new WhisperContext(context)
}

export async function releaseAllWhisper(): Promise<void> {
  if (!isJsiInstalled) return
  const { whisperReleaseAllContexts } = getJsi()
  return whisperReleaseAllContexts()
}

/** Current version of whisper.cpp */
export const libVersion: string = version

/** Is use CoreML models on iOS */
export const isUseCoreML: boolean = !!nativeConstants.useCoreML

/** Is allow fallback to CPU if load CoreML model failed */
export const isCoreMLAllowFallback: boolean =
  !!nativeConstants.coreMLAllowFallback

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

  gpu = false

  reasonNoGPU = ''

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
    const { whisperVadDetectSpeechFile } = getJsi()

    let path = ''
    if (typeof filePathOrBase64 === 'number') {
      path = resolvePathFromAsset(filePathOrBase64)
    } else if (filePathOrBase64.startsWith('data:audio/wav;base64,')) {
      path = filePathOrBase64
    } else {
      path = resolveLocalInputPath(
        filePathOrBase64,
        'VAD remote file is not supported, please download it first',
      )
    }

    const result = await whisperVadDetectSpeechFile(this.id, path, options)
    return result.segments || []
  }

  /**
   * Detect speech segments in raw audio data (base64 encoded float32 PCM data or ArrayBuffer)
   */
  async detectSpeechData(
    audioData: string | ArrayBuffer,
    options: VadOptions = {},
  ): Promise<VadSegment[]> {
    const { whisperVadDetectSpeech } = getJsi()
    const pcmData =
      audioData instanceof ArrayBuffer
        ? audioData
        : decodeBase64ToArrayBuffer(audioData)
    const result = await whisperVadDetectSpeech(this.id, options, pcmData)
    return result.segments || []
  }

  async release(): Promise<void> {
    const { whisperReleaseVadContext } = getJsi()
    return whisperReleaseVadContext(this.id)
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
  await installJsi()
  const { whisperInitVadContext } = getJsi()

  const path =
    typeof filePath === 'number'
      ? resolvePathFromAsset(filePath)
      : resolveLocalInputPath(
          filePath,
          'VAD remote file is not supported, please download it first',
        )

  const contextId = createContextId()
  const context = await whisperInitVadContext(contextId, {
    filePath: path,
    isBundleAsset: !!isBundleAsset,
    useGpu,
    nThreads,
  } satisfies NativeVadContextOptions)

  return new WhisperVadContext(context)
}

/**
 * Release all VAD contexts and free their memory
 * @returns Promise resolving when all contexts are released
 */
export async function releaseAllWhisperVad(): Promise<void> {
  if (!isJsiInstalled) return
  const { whisperReleaseAllVadContexts } = getJsi()
  return whisperReleaseAllVadContexts()
}

let logInitialized = false

/** Enable or disable native whisper.cpp logging */
export async function toggleNativeLog(enabled: boolean): Promise<void> {
  if (!enabled && !logInitialized) return

  logInitialized = true
  await installJsi()

  const { whisperToggleNativeLog } = getJsi()
  return whisperToggleNativeLog(enabled, enabled ? emitNativeLog : undefined)
}

/** Add a listener for native whisper.cpp log output */
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
