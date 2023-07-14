import React, { useCallback, useState } from 'react'
import {
  StyleSheet,
  ScrollView,
  View,
  Text,
  TouchableOpacity,
  SafeAreaView,
  Platform,
  PermissionsAndroid,
} from 'react-native'
import RNFS from 'react-native-fs'
// eslint-disable-next-line import/no-unresolved
import { initWhisper, libVersion } from 'whisper.rn'
import sampleFile from '../assets/jfk.wav'

if (Platform.OS === 'android') {
  // Request record audio permission
  PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO, {
    title: 'Whisper Audio Permission',
    message: 'Whisper needs access to your microphone',
    buttonNeutral: 'Ask Me Later',
    buttonNegative: 'Cancel',
    buttonPositive: 'OK',
  })
}

const styles = StyleSheet.create({
  container: {
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

function toTimestamp(t, comma = false) {
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

const mode = process.env.NODE_ENV === 'development' ? 'debug' : 'release'

const fileDir = `${RNFS.DocumentDirectoryPath}/whisper`

console.log('[App] fileDir', fileDir)

const modelFilePath = `${fileDir}/ggml-tiny.en.bin`

const createDir = async (log) => {
  if (!(await RNFS.exists(fileDir))) {
    log('Create dir', fileDir)
    await RNFS.mkdir(fileDir)
  }
}

const filterPath = (path) =>
  path.replace(RNFS.DocumentDirectoryPath, '<DocumentDir>')

export default function App() {
  const [whisperContext, setWhisperContext] = useState(null)
  const [logs, setLogs] = useState([`whisper.cpp version: ${libVersion}`])
  const [transcibeResult, setTranscibeResult] = useState(null)
  const [stopTranscribe, setStopTranscribe] = useState(null)

  const log = useCallback((...messages) => {
    setLogs((prev) => [...prev, messages.join(' ')])
  }, [])

  const progress = useCallback(
    ({ contentLength, bytesWritten }) => {
      const written = bytesWritten >= 0 ? bytesWritten : 0
      log(`Download progress: ${Math.round((written / contentLength) * 100)}%`)
    },
    [log],
  )

  return (
    <ScrollView contentInsetAdjustmentBehavior="automatic">
      <SafeAreaView style={styles.container}>
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
              log('Initialize context...')
              const startTime = Date.now()
              const ctx = await initWhisper({
                filePath: require('../assets/ggml-tiny.en.bin'),
                // If you want to use this option, please convert Core ML models by yourself
                // coreMLModelAsset:
                //   Platform.OS === 'ios'
                //     ? {
                //         filename: 'ggml-tiny.en-encoder.mlmodelc',
                //         assets: [
                //           require('../assets/ggml-tiny.en-encoder.mlmodelc/weights/weight.bin'),
                //           require('../assets/ggml-tiny.en-encoder.mlmodelc/model.mil'),
                //           require('../assets/ggml-tiny.en-encoder.mlmodelc/coremldata.bin'),
                //         ],
                //       }
                //     : undefined,
              })
              const endTime = Date.now()
              log('Loaded model, ID:', ctx.id)
              log('Loaded model in', endTime - startTime, `ms in ${mode} mode`)
              setWhisperContext(ctx)
            }}
          >
            <Text style={styles.buttonText}>Initialize (Use Asset)</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.button}
            onPress={async () => {
              if (whisperContext) {
                log('Found previous context')
                await whisperContext.release()
                setWhisperContext(null)
                log('Released previous context')
              }
              await createDir(log)
              if (await RNFS.exists(modelFilePath)) {
                log('Model already exists:')
                log(filterPath(modelFilePath))
              } else {
                log('Start Download Model to:')
                log(filterPath(modelFilePath))
                await RNFS.downloadFile({
                  fromUrl:
                    'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin',
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
            <Text style={styles.buttonText}>Initialize (Download)</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.buttons}>
          <TouchableOpacity
            style={styles.button}
            disabled={!!stopTranscribe?.stop}
            onPress={async () => {
              if (!whisperContext) return log('No context')

              log('Start transcribing...')
              const startTime = Date.now()
              const {
                // stop,
                promise,
              } = whisperContext.transcribe(sampleFile, {
                language: 'en',
                maxLen: 1,
                tokenTimestamps: true,
              })
              const { result, segments } = await promise
              const endTime = Date.now()
              setTranscibeResult(
                `Transcribed result: ${result}\n` +
                  `Transcribed in ${endTime - startTime}ms in ${mode} mode` +
                  `\n` +
                  `Segments:` +
                  `\n${segments
                    .map(
                      (segment) =>
                        `[${toTimestamp(segment.t0)} --> ${toTimestamp(
                          segment.t1,
                        )}]  ${segment.text}`,
                    )
                    .join('\n')}`,
              )
              log('Finished transcribing')
            }}
          >
            <Text style={styles.buttonText}>Transcribe File</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.button,
              stopTranscribe?.stop ? styles.buttonClear : null,
            ]}
            onPress={async () => {
              if (!whisperContext) return log('No context')
              if (stopTranscribe?.stop) {
                stopTranscribe?.stop()
                setStopTranscribe(null)
                return
              }
              log('Start realtime transcribing...')
              try {
                const { stop, subscribe } =
                  await whisperContext.transcribeRealtime({
                    language: 'en',
                    // Record duration in seconds
                    realtimeAudioSec: 60,
                    // Slice audio into 25 (or < 30) sec chunks for better performance
                    realtimeAudioSliceSec: 25,
                  })
                setStopTranscribe({ stop })
                subscribe((evt) => {
                  const { isCapturing, data, processTime, recordingTime } = evt
                  setTranscibeResult(
                    `Realtime transcribing: ${isCapturing ? 'ON' : 'OFF'}\n` +
                      `Result: ${data.result}\n\n` +
                      `Process time: ${processTime}ms\n` +
                      `Recording time: ${recordingTime}ms` +
                      `\n` +
                      `Segments:` +
                      `\n${data.segments
                        .map(
                          (segment) =>
                            `[${toTimestamp(segment.t0)} --> ${toTimestamp(
                              segment.t1,
                            )}]  ${segment.text}`,
                        )
                        .join('\n')}`,
                  )
                  if (!isCapturing) {
                    setStopTranscribe(null)
                    log('Finished realtime transcribing')
                  }
                })
              } catch (e) {
                log('Error:', e)
              }
            }}
          >
            <Text style={styles.buttonText}>
              {stopTranscribe?.stop ? 'Stop' : 'Realtime'}
            </Text>
          </TouchableOpacity>
        </View>
        <View style={styles.logContainer}>
          {logs.map((msg, index) => (
            <Text key={index} style={styles.logText}>
              {msg}
            </Text>
          ))}
        </View>
        {transcibeResult && (
          <View style={styles.logContainer}>
            <Text style={styles.logText}>{transcibeResult}</Text>
          </View>
        )}

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
        <TouchableOpacity
          style={[styles.button, styles.buttonClear]}
          onPress={() => {
            setLogs([])
            setTranscibeResult('')
          }}
        >
          <Text style={styles.buttonText}>Clear Logs</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.button, styles.buttonClear]}
          title="Clear Download files"
          onPress={async () => {
            await RNFS.unlink(fileDir).catch(() => {})
            log('Deleted files')
          }}
        >
          <Text style={styles.buttonText}>Clear Download files</Text>
        </TouchableOpacity>
      </SafeAreaView>
    </ScrollView>
  )
}
