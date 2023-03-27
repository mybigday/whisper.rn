import {
  NativeEventEmitter,
  DeviceEventEmitter,
  NativeModules,
  Platform,
  DeviceEventEmitterStatic,
} from 'react-native'

const LINKING_ERROR =
  `The package 'whisper.rn' doesn't seem to be linked. Make sure: \n\n${Platform.select({ ios: "- You have run 'pod install'\n", default: '' })
  }- You rebuilt the app after installing the package`

const RNWhisper = NativeModules.RNWhisper
  ? NativeModules.RNWhisper
  : new Proxy(
    {},
    {
      get() {
        throw new Error(LINKING_ERROR)
      },
    },
  )

let EventEmitter: NativeEventEmitter | DeviceEventEmitterStatic
if (Platform.OS === 'ios') {
  EventEmitter = new NativeEventEmitter(RNWhisper)
}
if (Platform.OS === 'android') {
  EventEmitter = DeviceEventEmitter
}

const EVENT_ON_REALTIME_TRANSCRIBE = '@RNWhisper_onRealtimeTranscribe'
const EVENT_ON_REALTIME_TRANSCRIBE_END = '@RNWhisper_onRealtimeTranscribeEnd'

export type TranscribeOptions = {
  /** Spoken language (Default: 'auto' for auto-detect) */
  language?: string,
  /** Translate from source language to english (Default: false) */
  translate?: boolean,
  /** Number of threads to use during computation (Default: 4) */
  maxThreads?: number,
  /** Maximum number of text context tokens to store */
  maxContext?: number,
  /** Maximum segment length in characters */
  maxLen?: number,
  /** Enable token-level timestamps */
  tokenTimestamps?: boolean,
  /** Word timestamp probability threshold */
  wordThold?: number,
  /** Time offset in milliseconds */
  offset?: number,
  /** Duration of audio to process in milliseconds */
  duration?: number,
  /** Tnitial decoding temperature */
  temperature?: number,
  temperatureInc?: number,
  /** Beam size for beam search */
  beamSize?: number,
  /** Number of best candidates to keep */
  bestOf?: number,
  /** Speed up audio by x2 (reduced accuracy) */
  speedUp?: boolean,
  /** Initial Prompt */
  prompt?: string,
}

export type TranscribeRealtimeOptions = TranscribeOptions & {
  /**
   * Realtime record max duration in seconds. 
   * Due to the whisper.cpp hard constraint - processes the audio in chunks of 30 seconds,
   * the recommended value will be <= 30 seconds. (Default: 30)
   */
  realtimeAudioSec?: number,
}

export type TranscribeResult = {
  result: string,
  segments: Array<{
    text: string,
    t0: number,
    t1: number,
  }>,
}

export type TranscribeRealtimeEvent = {
  contextId: number,
  jobId: number,
  /** Is capturing audio, when false, the event is the final result */
  isCapturing: boolean,
  code: number,
  processTime: number,
  recordingTime: number,
  data?: TranscribeResult,
  error?: string,
}

export type TranscribeRealtimeNativeEvent = {
  contextId: number,
  jobId: number,
  payload: {
    /** Is capturing audio, when false, the event is the final result */
    isCapturing: boolean,
    code: number,
    processTime: number,
    recordingTime: number,
    data?: TranscribeResult,
    error?: string,
  },
}

class WhisperContext {
  id: number

  constructor(id: number) {
    this.id = id
  }

  /** Transcribe audio file */
  transcribe(path: string, options: TranscribeOptions = {}): {
    /** Stop the transcribe */
    stop: () => void,
    /** Transcribe result promise */
    promise: Promise<TranscribeResult>,
  } {
    const jobId: number = Math.floor(Math.random() * 10000)
    return {
      stop: () => RNWhisper.abortTranscribe(this.id, jobId),
      promise: RNWhisper.transcribeFile(this.id, jobId, path, options),
    }
  }

  /** Transcribe the microphone audio stream, the microphone user permission is required */
  async transcribeRealtime(options: TranscribeRealtimeOptions = {}): Promise<{
    /** Stop the realtime transcribe */
    stop: () => void,
    /** Subscribe to realtime transcribe events */
    subscribe: (callback: (event: TranscribeRealtimeEvent) => void) => void,
  }> {
    const jobId: number = Math.floor(Math.random() * 10000)
    await RNWhisper.startRealtimeTranscribe(this.id, jobId, options)
    let removeTranscribe: () => void
    let removeEnd: () => void
    let lastTranscribePayload: TranscribeRealtimeNativeEvent['payload']
    return {
      stop: () => RNWhisper.abortTranscribe(this.id, jobId),
      subscribe: (callback: (event: TranscribeRealtimeEvent) => void) => {
        const transcribeListener = EventEmitter.addListener(
          EVENT_ON_REALTIME_TRANSCRIBE,
          (evt: TranscribeRealtimeNativeEvent) => {
            const { contextId, payload } = evt
            if (contextId !== this.id || evt.jobId !== jobId) return
            lastTranscribePayload = payload
            callback({ contextId, jobId: evt.jobId, ...payload })
            if (!payload.isCapturing) removeTranscribe()
          }
        )
        removeTranscribe = transcribeListener.remove
        const endListener = EventEmitter.addListener(
          EVENT_ON_REALTIME_TRANSCRIBE_END,
          (evt: TranscribeRealtimeNativeEvent) => {
            const { contextId } = evt
            if (contextId !== this.id || evt.jobId !== jobId) return
            callback({ contextId, jobId: evt.jobId, ...lastTranscribePayload, isCapturing: false })
            removeTranscribe?.()
            removeEnd()
          }
        )
        removeEnd = endListener.remove
      },
    }
  }

  async release() {
    return RNWhisper.releaseContext(this.id)
  }
}

export async function initWhisper(
  { filePath }: { filePath?: string } = {}
): Promise<WhisperContext> {
  const id = await RNWhisper.initContext(filePath)
  return new WhisperContext(id)
}

export async function releaseAllWhisper(): Promise<void> {
  return RNWhisper.releaseAllContexts()
}