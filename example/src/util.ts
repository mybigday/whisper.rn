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
