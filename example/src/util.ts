import RNFS from 'react-native-fs'

export const fileDir = `${RNFS.DocumentDirectoryPath}/whisper`

console.log('[App] fileDir', fileDir)

export const modelHost = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main'

export const createDir = async (log: any) => {
  if (!(await RNFS.exists(fileDir))) {
    log?.('Create dir', fileDir)
    await RNFS.mkdir(fileDir)
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
