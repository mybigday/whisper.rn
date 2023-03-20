import React, { useCallback, useState } from 'react'
import {
  StyleSheet,
  ScrollView,
  View,
  Text,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native'
import RNFS from 'react-native-fs'
// eslint-disable-next-line import/no-unresolved
import { initWhisper } from 'whisper.rn'

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttons: { flexDirection: 'row' },
  button: { margin: 4, backgroundColor: '#333', borderRadius: 4, padding: 8 },
  buttonClear: { backgroundColor: '#888' },
  buttonText: { fontSize: 14, color: 'white', textAlign: 'center' },
  logContainer: {
    backgroundColor: 'lightgray',
    padding: 8,
    width: '95%',
    borderRadius: 8,
    marginVertical: 8,
  },
  logText: { fontSize: 12, color: '#333' },
})

const mode = process.env.NODE_ENV === 'development' ? 'debug' : 'release'

const modelURL =
  'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin'
const sampleURL =
  'https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav'

const fileDir = `${RNFS.DocumentDirectoryPath}/whisper`

console.log('[App] fileDir', fileDir)

const modelFilePath = `${fileDir}/base.en`
const sampleFilePath = `${fileDir}/jfk.wav`

const filterPath = (path) =>
  path.replace(RNFS.DocumentDirectoryPath, '<DocumentDir>')

export default function App() {
  const [whisperContext, setWhisperContext] = useState(null)
  const [logs, setLogs] = useState([])

  const log = useCallback((...messages) => {
    setLogs((prev) => [...prev, messages.join(' ')])
  }, [])

  const createDir = useCallback(async () => {
    if (!(await RNFS.exists(fileDir))) {
      log('Create dir', fileDir)
      await RNFS.mkdir(fileDir)
    }
  }, [log])

  const progress = useCallback(
    ({ contentLength, bytesWritten }) => {
      const written = bytesWritten >= 0 ? bytesWritten : 0
      log(`Download progress: ${Math.round((written / contentLength) * 100)}%`)
    },
    [log],
  )

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.buttons}>
          <TouchableOpacity
            style={styles.button}
            onPress={async () => {
              if (whisperContext) {
                log('Found previous context')
                await whisperContext.release()
                setWhisperContext(null)
                log('Released previous context')
              }
              await createDir()
              if (await RNFS.exists(modelFilePath)) {
                log('Model already exists:')
                log(filterPath(modelFilePath))
              } else {
                log('Start Download Model to:')
                log(filterPath(modelFilePath))
                await RNFS.downloadFile({
                  fromUrl: modelURL,
                  toFile: modelFilePath,
                  progressInterval: 1000,
                  begin: () => {},
                  progress,
                }).promise
                log('Downloaded model file:')
                log(filterPath(modelFilePath))
              }
              log('Initialize context...')
              const startTime = Date.now()
              const ctx = await initWhisper({ filePath: modelFilePath })
              const endTime = Date.now()
              log('Loaded model, ID:', ctx.id)
              log('Loaded model in', endTime - startTime, `ms in ${mode} mode`)
              setWhisperContext(ctx)
            }}
          >
            <Text style={styles.buttonText}>Initialize Context</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.button}
            onPress={async () => {
              if (!whisperContext) {
                log('No context')
                return
              }
              await createDir()

              if (await RNFS.exists(sampleFilePath)) {
                log('Sample file already exists:')
                log(filterPath(sampleFilePath))
              } else {
                log('Start download sample file to:')
                log(filterPath(sampleFilePath))
                await RNFS.downloadFile({
                  fromUrl: sampleURL,
                  toFile: sampleFilePath,
                  progressInterval: 1000,
                  begin: () => {},
                  progress,
                }).promise
                log('Downloaded sample file:')
                log(filterPath(sampleFilePath))
              }
              log('Start transcribing...')
              const startTime = Date.now()
              const {
                // stop,
                promise
              } = whisperContext.transcribe(
                sampleFilePath,
                { language: 'en' },
              ).catch(e => {
                log(e.message)
                return null
              })
              const { result } = await promise
              const endTime = Date.now()
              log('Transcribed result:', result)
              log('Transcribed in', endTime - startTime, `ms in ${mode} mode`)
            }}
          >
            <Text style={styles.buttonText}>Transcribe</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.buttonClear]}
            onPress={async () => {
              if (!whisperContext) return
              await whisperContext.release()
              setWhisperContext(null)
              log('Released context')
            }}
          >
            <Text style={styles.buttonText}>Release Context</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.logContainer}>
          {logs.map((msg, index) => (
            <Text key={index} style={styles.logText}>
              {msg}
            </Text>
          ))}
        </View>
        <TouchableOpacity
          style={[styles.button, styles.buttonClear]}
          onPress={() => setLogs([])}
        >
          <Text style={styles.buttonText}>Clear Logs</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.button, styles.buttonClear]}
          title="Clear Download files"
          onPress={async () => {
            await RNFS.unlink(modelFilePath).catch(() => {})
            await RNFS.unlink(sampleFilePath).catch(() => {})
            log('Deleted files')
          }}
        >
          <Text style={styles.buttonText}>Clear Download files</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  )
}
