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
import { unzip } from 'react-native-zip-archive'
import Sound from 'react-native-sound'
import { initWhisper, libVersion, AudioSessionIos } from '../../src' // whisper.rn
import type { WhisperContext } from '../../src'
import contextOpts from './context-opts'

const sampleFile = require('../assets/jfk.wav')

if (Platform.OS === 'android') {
  // Request record audio permission
  // @ts-ignore
  PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO, { 
    title: 'Whisper Audio Permission',
    message: 'Whisper needs access to your microphone',
    buttonNeutral: 'Ask Me Later',
    buttonNegative: 'Cancel',
    buttonPositive: 'OK',
  })
}

const styles = StyleSheet.create({
  scrollview: { flexGrow: 1, justifyContent: 'center' },
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 4,
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

function toTimestamp(t: number, comma = false) {
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

const recordFile = `${fileDir}/realtime.wav`

const modelHost = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main'

const createDir = async (log: any) => {
  if (!(await RNFS.exists(fileDir))) {
    log('Create dir', fileDir)
    await RNFS.mkdir(fileDir)
  }
}

const filterPath = (path: string) =>
  path.replace(RNFS.DocumentDirectoryPath, '<DocumentDir>')

const updateAudioSession = async (log: any) => {
  if (Platform.OS !== 'ios') return

  // Log current audio session
  // log('Category & Options:', JSON.stringify(await AudioSessionIos.getCurrentCategory()))
  // log('Mode:', await AudioSessionIos.getCurrentMode())

  await AudioSessionIos.setCategory(
    AudioSessionIos.Category.PlayAndRecord, [
      AudioSessionIos.CategoryOption.MixWithOthers,
      AudioSessionIos.CategoryOption.AllowBluetooth,
    ],
  )
  await AudioSessionIos.setMode(AudioSessionIos.Mode.Default)
  await AudioSessionIos.setActive(true)

  const categoryResult = await AudioSessionIos.getCurrentCategory()
  log('Category:', categoryResult.category)
  log('Category Options:', categoryResult.options.join(', '))
  log('Mode:', await AudioSessionIos.getCurrentMode())
}

export default function App() {
  const [whisperContext, setWhisperContext] = useState<WhisperContext | null>(null)
  const [logs, setLogs] = useState([`whisper.cpp version: ${libVersion}`])
  const [transcibeResult, setTranscibeResult] = useState<string | null>(null)
  const [stopTranscribe, setStopTranscribe] = useState<{ stop: () => void } | null>(null)

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
    <ScrollView
      contentInsetAdjustmentBehavior="automatic"
      contentContainerStyle={styles.scrollview}
    >
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
                ...contextOpts,
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
              const modelFilePath = `${fileDir}/ggml-tiny.en.bin`
              if (await RNFS.exists(modelFilePath)) {
                log('Model already exists:')
                log(filterPath(modelFilePath))
              } else {
                log('Start Download Model to:')
                log(filterPath(modelFilePath))
                await RNFS.downloadFile({
                  fromUrl: `${modelHost}/ggml-tiny.en.bin`,
                  toFile: modelFilePath,
                  progressInterval: 1000,
                  begin: () => {},
                  progress,
                }).promise
                log('Downloaded model file:')
                log(filterPath(modelFilePath))
              }

              // If you don't want to enable Core ML, you can remove this
              const coremlModelFilePath = `${fileDir}/ggml-tiny.en-encoder.mlmodelc.zip`
              if (
                Platform.OS === 'ios' &&
                (await RNFS.exists(coremlModelFilePath))
              ) {
                log('Core ML Model already exists:')
                log(filterPath(coremlModelFilePath))
              } else if (Platform.OS === 'ios') {
                log('Start Download Core ML Model to:')
                log(filterPath(coremlModelFilePath))
                await RNFS.downloadFile({
                  fromUrl: `${modelHost}/ggml-tiny.en-encoder.mlmodelc.zip`,
                  toFile: coremlModelFilePath,
                  progressInterval: 1000,
                  begin: () => {},
                  progress,
                }).promise
                log('Downloaded Core ML Model model file:')
                log(filterPath(modelFilePath))
                await unzip(coremlModelFilePath, fileDir)
                log('Unzipped Core ML Model model successfully.')
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
              const { stop, promise } = whisperContext.transcribe(sampleFile, {
                language: 'en',
                maxLen: 1,
                tokenTimestamps: true,
                onProgress: (cur) => {
                  log(`Transcribing progress: ${cur}%`)
                },
              })
              setStopTranscribe({ stop })
              const { result, segments } = await promise
              const endTime = Date.now()
              setStopTranscribe(null)
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
                const t0 = Date.now()
                await stopTranscribe?.stop()
                const t1 = Date.now()
                log('Stopped transcribing in', t1 - t0, 'ms')
                setStopTranscribe(null)
                return
              }
              log('Start realtime transcribing...')
              try {
                await createDir(log)
                await updateAudioSession(log)
                const { stop, subscribe } =
                  await whisperContext.transcribeRealtime({
                    language: 'en',
                    // Record duration in seconds
                    realtimeAudioSec: 60,
                    // Slice audio into 25 (or < 30) sec chunks for better performance
                    realtimeAudioSliceSec: 25,
                    // Save audio on stop
                    audioOutputPath: recordFile,
                    // Voice Activity Detection - Start transcribing when speech is detected
                    // useVad: true,
                  })
                setStopTranscribe({ stop })
                subscribe((evt) => {
                  const { isCapturing, data, processTime, recordingTime } = evt
                  setTranscibeResult(
                    `Realtime transcribing: ${isCapturing ? 'ON' : 'OFF'}\n` +
                      `Result: ${data?.result}\n\n` +
                      `Process time: ${processTime}ms\n` +
                      `Recording time: ${recordingTime}ms` +
                      `\n` +
                      `Segments:` +
                      `\n${data?.segments
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
          onPress={async () => {
            await RNFS.unlink(fileDir).catch(() => {})
            log('Deleted files')
          }}
        >
          <Text style={styles.buttonText}>Clear Download files</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.button, styles.buttonClear]}
          onPress={async () => {
            if (!await RNFS.exists(recordFile)) {
              log('Recorded file does not exist')
              return
            }
            await updateAudioSession(log)
            const player = new Sound(recordFile, '', (e) => {
              if (e) {
                log('error', e)
                return
              }
              player.play((success) => {
                if (success) {
                  log('successfully finished playing');
                } else {
                  log('playback failed due to audio decoding errors');
                }
                player.release();
              });
            })
          }}
        >
          <Text style={styles.buttonText}>Play Recorded file</Text>
        </TouchableOpacity>
      </SafeAreaView>
    </ScrollView>
  )
}
