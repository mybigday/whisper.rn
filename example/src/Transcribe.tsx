import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  StyleSheet,
  ScrollView,
  View,
  Text,
  Platform,
  PermissionsAndroid,
} from 'react-native'
import RNFS from 'react-native-fs'
import Sound from 'react-native-sound'
import { initWhisper, libVersion, AudioSessionIos } from '../../src' // whisper.rn
import type { WhisperContext } from '../../src'
import { Button } from './Button'
import contextOpts from './context-opts'
import {
  createDir,
  fileDir,
  toTimestamp,
  downloadModel,
  whisperModels,
  WhisperModel,
} from './util'

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
  buttons: { flexDirection: 'row', margin: 8 },
  button: { margin: 4, backgroundColor: '#333', borderRadius: 4, padding: 8 },
  buttonClear: { backgroundColor: '#888' },
  buttonText: { fontSize: 14, color: 'white', textAlign: 'center' },
  configTitle: { fontSize: 16, fontWeight: 'bold', textAlign: 'center' },
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

const recordFile = `${fileDir}/realtime.wav`

export default function App() {
  const whisperContextRef = useRef<WhisperContext | null>(null)
  const whisperContext = whisperContextRef.current
  const [logs, setLogs] = useState([`whisper.cpp version: ${libVersion}`])
  const [transcibeResult, setTranscibeResult] = useState<string | null>(null)
  const [stopTranscribe, setStopTranscribe] = useState<{
    stop: () => void
  } | null>(null)
  const [selectedModel, setSelectedModel] = useState<WhisperModel>('base')
  const [downloadProgress, setDownloadProgress] = useState<number>(0)

  const log = useCallback((...messages: any[]) => {
    setLogs((prev) => [...prev, messages.join(' ')])
  }, [])

  useEffect(
    () => () => {
      whisperContextRef.current?.release()
      whisperContextRef.current = null
    },
    [],
  )

  return (
    <ScrollView
      contentInsetAdjustmentBehavior="automatic"
      contentContainerStyle={styles.scrollview}
    >
      <View style={styles.container}>
        <Text style={styles.configTitle}>Transcribe File Demo</Text>
        <View style={styles.buttons}>
          <Button
            title="Initialize (Use Asset base.bin)"
            onPress={async () => {
              if (whisperContext) {
                log('Found previous context')
                await whisperContext.release()
                whisperContextRef.current = null
                log('Released previous context')
              }
              log('Initialize context...')
              const startTime = Date.now()
              const ctx = await initWhisper({
                filePath: require('../assets/ggml-base.bin'),
                ...contextOpts,
              })
              const endTime = Date.now()
              log('Loaded model, ID:', ctx.id)
              log('Loaded model in', endTime - startTime, `ms in ${mode} mode`)
              whisperContextRef.current = ctx
            }}
          />
        </View>
        <Text style={styles.configTitle}>Whisper Model Selection</Text>
        <View style={styles.buttons}>
          {whisperModels.map((model) => (
            <Button
              key={model}
              title={model}
              style={[
                selectedModel === model ? { backgroundColor: '#007AFF' } : null,
              ]}
              onPress={() => setSelectedModel(model)}
            />
          ))}
        </View>
        <Button
          title={`Download & Initialize ${selectedModel}`}
          onPress={async () => {
            if (whisperContext) {
              log('Found previous context')
              await whisperContext.release()
              whisperContextRef.current = null
              log('Released previous context')
            }

            try {
              const modelFilePath = await downloadModel(
                selectedModel,
                (downloadProgressValue) => {
                  setDownloadProgress(downloadProgressValue)
                  log(
                    `Download progress: ${Math.round(
                      downloadProgressValue * 100,
                    )}%`,
                  )
                },
                log,
              )

              log('Initialize context...')
              const startTime = Date.now()
              const ctx = await initWhisper({ filePath: modelFilePath })
              const endTime = Date.now()
              log('Loaded model, ID:', ctx.id)
              log('Loaded model in', endTime - startTime, `ms in ${mode} mode`)
              whisperContextRef.current = ctx
              setDownloadProgress(0)
            } catch (error) {
              log('Error downloading or initializing model:', error)
              setDownloadProgress(0)
            }
          }}
        />
        {downloadProgress > 0 && downloadProgress < 1 && (
          <View style={styles.logContainer}>
            <Text style={styles.logText}>
              {`Downloading ${selectedModel}: ${Math.round(downloadProgress * 100)}%`}
            </Text>
          </View>
        )}
        <View style={styles.buttons}>
          <Button
            title="Transcribe File"
            disabled={!!stopTranscribe?.stop}
            onPress={async () => {
              if (!whisperContext) return log('No context')

              log('Start transcribing...')
              const startTime = Date.now()
              const { stop, promise } = whisperContext.transcribe(sampleFile, {
                maxLen: 1,
                tokenTimestamps: true,
                onProgress: (cur) => {
                  log(`Transcribing progress: ${cur}%`)
                },
                language: 'en',
                // prompt: 'HELLO WORLD',
                // onNewSegments: (segments) => {
                //   console.log('New segments:', segments)
                // },
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
          />
          <Button
            title={stopTranscribe?.stop ? 'Stop' : 'Realtime'}
            style={[stopTranscribe?.stop ? styles.buttonClear : null]}
            onPress={async () => {
              if (!whisperContext) return log('No context')
              if (stopTranscribe?.stop) {
                const t0 = Date.now()
                stopTranscribe?.stop()
                const t1 = Date.now()
                log('Stopped transcribing in', t1 - t0, 'ms')
                setStopTranscribe(null)
                return
              }
              log('Start realtime transcribing...')
              try {
                await createDir(log)
                const { stop, subscribe } =
                  await whisperContext.transcribeRealtime({
                    maxLen: 1,
                    language: 'en',
                    // Enable beam search (may be slower than greedy but more accurate)
                    // beamSize: 2,
                    // Record duration in seconds
                    realtimeAudioSec: 60,
                    // Slice audio into 25 (or < 30) sec chunks for better performance
                    realtimeAudioSliceSec: 25,
                    // Save audio on stop
                    audioOutputPath: recordFile,
                    // iOS Audio Session
                    audioSessionOnStartIos: {
                      category: AudioSessionIos.Category.PlayAndRecord,
                      options: [
                        AudioSessionIos.CategoryOption.MixWithOthers,
                        AudioSessionIos.CategoryOption.AllowBluetooth,
                      ],
                      mode: AudioSessionIos.Mode.Default,
                    },
                    audioSessionOnStopIos: 'restore', // Or an AudioSessionSettingIos
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
          />
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
        <Button
          title="Release Context"
          style={styles.buttonClear}
          onPress={async () => {
            if (!whisperContext) return
            await whisperContext.release()
            whisperContextRef.current = null
            log('Released context')
          }}
        />
        <Button
          title="Clear Logs"
          style={styles.buttonClear}
          onPress={() => {
            setLogs([])
            setTranscibeResult('')
          }}
        />
        <Button
          title="Clear Download files"
          style={styles.buttonClear}
          onPress={async () => {
            await RNFS.unlink(fileDir).catch(() => {})
            log('Deleted files')
          }}
        />
        <Button
          title="Play Recorded file"
          style={styles.buttonClear}
          onPress={async () => {
            if (!(await RNFS.exists(recordFile))) {
              log('Recorded file does not exist')
              return
            }
            const player = new Sound(recordFile, '', (e) => {
              if (e) {
                log('error', e)
                return
              }
              player.play((success) => {
                if (success) {
                  log('successfully finished playing')
                } else {
                  log('playback failed due to audio decoding errors')
                }
                player.release()
              })
            })
          }}
        />
      </View>
    </ScrollView>
  )
}
