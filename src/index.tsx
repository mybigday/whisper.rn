import { NativeModules, Platform } from 'react-native'

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

export type TranscribeOptions = {
  abortControl?: {
    contextId: number,
    jobId: number,
    abort: () => void,
  },
  maxThreads?: number,
  maxContext?: number,
  maxLen?: number,
  offset?: number,
  duration?: number,
  wordThold?: number,
  temperature?: number,
  temperatureInc?: number,
  beamSize?: number,
  bestOf?: number,
  speedUp?: boolean,
}

export type TranscribeResult = {
  result: string,
}

class WhisperContext {
  id: number

  constructor(id: number) {
    this.id = id
  }

  async transcribe(path: string, options: TranscribeOptions = {}): Promise<TranscribeResult> {
    const jobId: number = Math.floor(Math.random() * 10000)
    if (options.abortControl) {
      options.abortControl.contextId = this.id
      options.abortControl.jobId = jobId
    }
    return RNWhisper.transcribe(this.id, jobId, path, options).then((result: string) => ({
      result
    }))
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

export const createAbortControl = () => ({
  contextId: null,
  jobId: null,
  abort() {
    if (!this.jobId) return
    RNWhisper.abortTranscribe(this.contextId, this.jobId)
  }
})

export async function releaseAllWhisper(): Promise<void> {
  return RNWhisper.releaseAllContexts()
}