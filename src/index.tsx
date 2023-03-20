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

  transcribe(path: string, options: TranscribeOptions = {}): {
    stop: () => void,
    promise: Promise<TranscribeResult>,
  } {
    const jobId: number = Math.floor(Math.random() * 10000)
    return {
      stop: () => RNWhisper.abortTranscribe(this.id, jobId),
      promise: RNWhisper.transcribe(this.id, jobId, path, options).then((result: string) => ({
        result
      }))
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