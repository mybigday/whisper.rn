import RNFS from 'react-native-fs'

export const fileDir = `${RNFS.DocumentDirectoryPath}/whisper`

console.log('[App] fileDir', fileDir)

export const modelHost = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main'

export const vadModelHost = 'https://huggingface.co/ggml-org/whisper-vad/resolve/main'

export const createDir = async (log: any) => {
  if (!(await RNFS.exists(fileDir))) {
    log?.('Create dir', fileDir)
    await RNFS.mkdir(fileDir)
  }
}

export const whisperModels = [
  'tiny',
  'tiny.en',
  'base',
  'base.en',
  'small',
  'small.en',
  'medium',
  'large-v3',
  'large-v3-turbo'
] as const

export type WhisperModel = typeof whisperModels[number]

export const downloadModel = async (
  model: WhisperModel,
  onProgress?: (progress: number) => void,
  log?: (message: string) => void
): Promise<string> => {
  const modelFileName = `ggml-${model}.bin`
  const modelPath = `${fileDir}/${modelFileName}`
  const modelUrl = `${modelHost}/${modelFileName}`

  await createDir(log)

  if (await RNFS.exists(modelPath)) {
    log?.(`Model ${model} already exists at ${modelPath}`)
    return modelPath
  }

  log?.(`Downloading ${model} model from ${modelUrl}`)

  const downloadOptions = {
    fromUrl: modelUrl,
    toFile: modelPath,
    progress: onProgress ? (res: any) => {
      const progress = res.bytesWritten / res.contentLength
      onProgress(progress)
    } : undefined
  }

  try {
    await RNFS.downloadFile(downloadOptions).promise
    log?.(`Successfully downloaded ${model} model to ${modelPath}`)
    return modelPath
  } catch (error) {
    log?.(`Failed to download ${model} model: ${error}`)
    if (await RNFS.exists(modelPath)) {
      await RNFS.unlink(modelPath)
    }
    throw error
  }
}

export function toTimestamp(t: number, comma = false) {
  let msec = t * 10
  const hr = Math.floor(msec / (1000 * 60 * 60))
  msec -= hr * (1000 * 60 * 60)
  const min = Math.floor(msec / (1000 * 60))
  msec -= min * (1000 * 60)
  const sec = Math.floor(msec / 1000)
  msec -= sec * 1000

  const separator = comma ? ',' : '.'
  const timestamp = `${String(hr).padStart(2, '0')}:${String(min).padStart(
    2,
    '0',
  )}:${String(sec).padStart(2, '0')}${separator}${String(msec).padStart(
    3,
    '0',
  )}`

  return timestamp
}
