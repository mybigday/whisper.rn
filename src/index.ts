import {
  NativeEventEmitter,
  DeviceEventEmitter,
  Platform,
  DeviceEventEmitterStatic,
  Image,
} from 'react-native'
import RNWhisper from './NativeRNWhisper'
import type {
  TranscribeOptions,
  TranscribeResult,
  CoreMLAsset,
} from './NativeRNWhisper'
import { version } from './version.json'

let EventEmitter: NativeEventEmitter | DeviceEventEmitterStatic
if (Platform.OS === 'ios') {
  // @ts-ignore
  EventEmitter = new NativeEventEmitter(RNWhisper)
}
if (Platform.OS === 'android') {
  EventEmitter = DeviceEventEmitter
}

export type { TranscribeOptions, TranscribeResult }

const EVENT_ON_REALTIME_TRANSCRIBE = '@RNWhisper_onRealtimeTranscribe'
const EVENT_ON_REALTIME_TRANSCRIBE_END = '@RNWhisper_onRealtimeTranscribeEnd'

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

export class WhisperContext {
  id: number

  constructor(id: number) {
    this.id = id
  }

  /** Transcribe audio file */
  transcribe(
    filePath: string | number,
    options: TranscribeOptions = {},
  ): {
    /** Stop the transcribe */
    stop: () => void
    /** Transcribe result promise */
    promise: Promise<TranscribeResult>
  } {
    let path = ''
    if (typeof filePath === 'number') {
      try {
        const source = Image.resolveAssetSource(filePath)
        if (source) path = source.uri
      } catch (e) {
        throw new Error(`Invalid asset: ${filePath}`)
      }
    } else {
      if (filePath.startsWith('http'))
        throw new Error('Transcribe remote file is not supported, please download it first')
      path = filePath
    }
    const jobId: number = Math.floor(Math.random() * 10000)
    return {
      stop: () => RNWhisper.abortTranscribe(this.id, jobId),
      promise: RNWhisper.transcribeFile(this.id, jobId, path, options),
    }
  }

  /** Transcribe the microphone audio stream, the microphone user permission is required */
  async transcribeRealtime(options: TranscribeRealtimeOptions = {}): Promise<{
    /** Stop the realtime transcribe */
    stop: () => void
    /** Subscribe to realtime transcribe events */
    subscribe: (callback: (event: TranscribeRealtimeEvent) => void) => void
  }> {
    const jobId: number = Math.floor(Math.random() * 10000)
    await RNWhisper.startRealtimeTranscribe(this.id, jobId, options)
    let lastTranscribePayload: TranscribeRealtimeNativePayload

    const slices: TranscribeRealtimeNativePayload[] = []
    let sliceIndex: number = 0
    let tOffset: number = 0

    const putSlice = (payload: TranscribeRealtimeNativePayload) => {
      if (!payload.isUseSlices) return
      if (sliceIndex !== payload.sliceIndex) {
        const { segments = [] } = slices[sliceIndex]?.data || {}
        tOffset = segments[segments.length - 1]?.t1 || 0
      }
      ;({ sliceIndex } = payload)
      slices[sliceIndex] = {
        ...payload,
        data: payload.data
          ? {
              ...payload.data,
              segments:
                payload.data.segments.map((segment) => ({
                  ...segment,
                  t0: segment.t0 + tOffset,
                  t1: segment.t1 + tOffset,
                })) || [],
            }
          : undefined,
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

    return {
      stop: () => RNWhisper.abortTranscribe(this.id, jobId),
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
  coreMLModelAssets?: {
    filename: string
    assets: number[]
  }
  /** Is the file path a bundle asset for pure string filePath */
  isBundleAsset?: boolean
}

const coreMLModelAssetPaths = [
  'weights/weight.bin',
  'model.mil',
  'coremldata.bin',
]

export async function initWhisper({
  filePath,
  coreMLModelAssets,
  isBundleAsset,
}: ContextOptions): Promise<WhisperContext> {
  let path = ''
  let coreMLAssets: CoreMLAsset[] | undefined
  if (coreMLModelAssets) {
    const { filename, assets } = coreMLModelAssets
    if (filename && assets) {
      coreMLAssets = assets
        ?.map((asset) => {
          const { uri } = Image.resolveAssetSource(asset)
          const filepath = coreMLModelAssetPaths.find((p) => uri.includes(p))
          if (filepath) {
            return {
              uri,
              filepath: `${filename}/${filepath}`,
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
        isBundleAsset = false
      }
    } catch (e) {
      throw new Error(`Invalid asset: ${filePath}`)
    }
  } else {
    if (filePath.startsWith('http'))
      throw new Error('Transcribe remote file is not supported, please download it first')
    path = filePath
  }
  const id = await RNWhisper.initContext({
    filePath: path,
    isBundleAsset: !!isBundleAsset,
    // Only development mode need download Core ML model assets (from packager server)
    downloadCoreMLAssets: __DEV__ && !!coreMLAssets,
    coreMLAssets,
  })
  return new WhisperContext(id)
}

export async function releaseAllWhisper(): Promise<void> {
  return RNWhisper.releaseAllContexts()
}

/** Current version of whisper.cpp */
export const libVersion: string = version

const { useCoreML, coreMLAllowFallback } = RNWhisper.getConstants?.() || {}

/** Is use CoreML models on iOS */
export const isUseCoreML: boolean = !!useCoreML

/** Is allow fallback to CPU if load CoreML model failed */
export const isCoreMLAllowFallback: boolean = !!coreMLAllowFallback
