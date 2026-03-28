import React, { useCallback, useEffect, useRef, useState } from 'react'
import { StyleSheet, ScrollView, View, Text } from 'react-native'
import RNFS from 'react-native-fs'
import Sound from 'react-native-sound'
import LiveAudioStream from '@fugood/react-native-audio-pcm-stream'
import { Buffer } from 'buffer'
import { initWhisperVad, libVersion } from '../../src' // whisper.rn
import type { WhisperVadContext, VadSegment } from '../../src'
import { Button } from './Button'
import { createDir, fileDir, vadModelHost, toTimestamp } from './util'
import { WavFileWriter } from '../../src/utils/WavFileWriter'

const sampleFile = require('../assets/jfk.wav')

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

const mode = process.env.NODE_ENV === 'development' ? 'debug' : 'release'
const recordFile = `${fileDir}/vad-record.wav`

const audioOptions = {
  sampleRate: 16000,
  channels: 1,
  bitsPerSample: 16,
  audioSource: 6,
  wavFile: recordFile,
  bufferSize: 16 * 1024,
}

const filterPath = (path: string) =>
  path.replace(RNFS.DocumentDirectoryPath, '<DocumentDir>')

export default function VadExample() {
  const vadContextRef = useRef<WhisperVadContext | null>(null)
  const vadContext = vadContextRef.current
  const [logs, setLogs] = useState([
    `whisper.cpp version: ${libVersion}`,
    'VAD Example - Voice Activity Detection',
  ])
  const [vadResult, setVadResult] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const recordedDataRef = useRef<Uint8Array | null>(null)

  const log = useCallback((...messages: any[]) => {
    setLogs((prev) => [...prev, messages.join(' ')])
  }, [])

  useEffect(
    () => () => {
      vadContextRef.current?.release()
      vadContextRef.current = null
    },
    [],
  )

  const progress = useCallback(
    ({
      contentLength,
      bytesWritten,
    }: {
      contentLength: number
      bytesWritten: number
    }) => {
      const written = bytesWritten >= 0 ? bytesWritten : 0
      log(`Download progress: ${Math.round((written / contentLength) * 100)}%`)
    },
    [log],
  )

  const startRecording = async () => {
    try {
      await createDir(log)
      recordedDataRef.current = null

      LiveAudioStream.init(audioOptions)
      LiveAudioStream.on('data', (data: string) => {
        const newData = new Uint8Array(Buffer.from(data, 'base64'))
        if (!recordedDataRef.current) {
          recordedDataRef.current = newData
        } else {
          const combined = new Uint8Array(
            recordedDataRef.current.length + newData.length,
          )
          combined.set(recordedDataRef.current)
          combined.set(newData, recordedDataRef.current.length)
          recordedDataRef.current = combined
        }
      })

      LiveAudioStream.start()
      setIsRecording(true)
      log('Started recording for VAD analysis...')
    } catch (error) {
      log('Error starting recording:', error)
    }
  }

  const stopRecording = async () => {
    try {
      await LiveAudioStream.stop()
      setIsRecording(false)
      log('Stopped recording')

      if (!recordedDataRef.current) {
        log('No recorded data')
        return
      }

      // Save the recorded data as WAV file for playback
      const wavFileWriter = new WavFileWriter(RNFS, recordFile, audioOptions)
      await wavFileWriter.initialize()
      await wavFileWriter.appendAudioData(recordedDataRef.current!)
      await wavFileWriter.finalize()

      log(`Recorded ${recordedDataRef.current.length} bytes of audio data`)
    } catch (error) {
      log('Error stopping recording:', error)
    }
  }

  return (
    <ScrollView
      contentInsetAdjustmentBehavior="automatic"
      contentContainerStyle={styles.scrollview}
    >
      <View style={styles.container}>
        <View style={styles.buttons}>
          <Button
            title="Initialize VAD (Use Asset)"
            onPress={async () => {
              if (vadContext) {
                log('Found previous VAD context')
                await vadContext.release()
                vadContextRef.current = null
                log('Released previous VAD context')
              }
              log('Initialize VAD context...')
              const startTime = Date.now()
              const ctx = await initWhisperVad({
                filePath: require('../assets/ggml-silero-v6.2.0.bin'),
                useGpu: true,
                nThreads: 4,
              })
              const endTime = Date.now()
              log('Loaded VAD model, ID:', ctx.id)
              log(
                `Loaded VAD model in ${endTime - startTime}ms in ${mode} mode`,
              )
              vadContextRef.current = ctx
            }}
          />
          <Button
            title="Initialize VAD (Download)"
            onPress={async () => {
              if (vadContext) {
                log('Found previous VAD context')
                await vadContext.release()
                vadContextRef.current = null
                log('Released previous VAD context')
              }
              await createDir(log)
              const modelFilePath = `${fileDir}/ggml-silero-v6.2.0.bin`
              if (await RNFS.exists(modelFilePath)) {
                log('Model already exists:')
                log(filterPath(modelFilePath))
              } else {
                log('Start Download Model for VAD to:')
                log(filterPath(modelFilePath))
                await RNFS.downloadFile({
                  fromUrl: `${vadModelHost}/ggml-silero-v6.2.0.bin`,
                  toFile: modelFilePath,
                  progressInterval: 1000,
                  begin: () => {},
                  progress,
                }).promise
                log('Downloaded VAD model file:')
                log(filterPath(modelFilePath))
              }

              log('Initialize VAD context...')
              const startTime = Date.now()
              const ctx = await initWhisperVad({
                filePath: modelFilePath,
                useGpu: true,
                nThreads: 4,
              })
              const endTime = Date.now()
              log('Loaded VAD model, ID:', ctx.id)
              log(
                `Loaded VAD model in ${endTime - startTime}ms in ${mode} mode`,
              )
              vadContextRef.current = ctx
            }}
          />
        </View>
        <View style={styles.buttons}>
          <Button
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
            onPress={isRecording ? stopRecording : startRecording}
            disabled={!vadContext}
          />
          <Button
            title="Detect Speech (Recorded Data)"
            onPress={async () => {
              if (!vadContext) return log('No VAD context')
              if (!recordedDataRef.current)
                return log('No recorded data available')

              log('Start VAD detection on recorded data...')
              const startTime = Date.now()

              // Convert recorded data to base64
              const base64Data = Buffer.from(recordedDataRef.current).toString(
                'base64',
              )

              const segments: VadSegment[] = await vadContext.detectSpeechData(
                base64Data,
                {
                  threshold: 0.5,
                  minSpeechDurationMs: 250,
                  minSilenceDurationMs: 100,
                  maxSpeechDurationS: 30,
                  speechPadMs: 30,
                  samplesOverlap: 0.1,
                },
              )

              const endTime = Date.now()

              if (segments.length === 0) {
                setVadResult('No speech segments detected in recorded data')
                log('No speech segments found in recorded data')
              } else {
                const resultText = `Recorded Data - Detected ${
                  segments.length
                } speech segments:
Detection time: ${endTime - startTime}ms in ${mode} mode

Speech Segments:
${segments
  .map(
    (segment, index) =>
      `${index + 1}. [${toTimestamp(
        Math.round(segment.t0),
      )} --> ${toTimestamp(Math.round(segment.t1))}] Duration: ${(
        (segment.t1 - segment.t0) / 100
      ).toFixed(2)}s`,
  )
  .join('\n')}`

                setVadResult(resultText)
                log(
                  `Detected ${segments.length} speech segments in recorded data`,
                )
              }
            }}
            disabled={!vadContext || !recordedDataRef.current}
          />
        </View>
        <View style={styles.buttons}>
          <Button
            title="Detect Speech (Default)"
            onPress={async () => {
              if (!vadContext) return log('No VAD context')

              log('Start VAD detection with default settings...')
              const startTime = Date.now()

              // Use actual sample audio file for VAD detection
              // Now supports same formats as transcribe: files, URLs, base64 WAV, assets
              const segments: VadSegment[] = await vadContext.detectSpeech(
                sampleFile,
                {
                  threshold: 0.5,
                  minSpeechDurationMs: 250,
                  minSilenceDurationMs: 100,
                  maxSpeechDurationS: 30,
                  speechPadMs: 30,
                  samplesOverlap: 0.1,
                },
              )

              const endTime = Date.now()

              if (segments.length === 0) {
                setVadResult('No speech segments detected')
                log('No speech segments found')
              } else {
                const resultText = `Detected ${segments.length} speech segments:
Detection time: ${endTime - startTime}ms in ${mode} mode

Speech Segments:
${segments
  .map(
    (segment, index) =>
      `${index + 1}. [${toTimestamp(
        Math.round(segment.t0),
      )} --> ${toTimestamp(Math.round(segment.t1))}] Duration: ${(
        (segment.t1 - segment.t0) / 100
      ).toFixed(2)}s`,
  )
  .join('\n')}`

                setVadResult(resultText)
                log(`Detected ${segments.length} speech segments`)
              }
            }}
          />
          <Button
            title="Detect Speech (Sensitive)"
            onPress={async () => {
              if (!vadContext) return log('No VAD context')

              log('Start VAD detection with sensitive settings...')
              const startTime = Date.now()

              const segments: VadSegment[] = await vadContext.detectSpeech(
                sampleFile,
                {
                  threshold: 0.3, // More sensitive
                  minSpeechDurationMs: 100, // Shorter minimum duration
                  minSilenceDurationMs: 50, // Shorter silence requirement
                  maxSpeechDurationS: 15, // Shorter max segments
                  speechPadMs: 50, // More padding
                  samplesOverlap: 0.2, // More overlap
                },
              )

              const endTime = Date.now()

              if (segments.length === 0) {
                setVadResult('No speech segments detected (sensitive mode)')
                log('No speech segments found (sensitive)')
              } else {
                const resultText = `Sensitive Detection - Found ${
                  segments.length
                } speech segments:
Detection time: ${endTime - startTime}ms in ${mode} mode

Speech Segments:
${segments
  .map(
    (segment, index) =>
      `${index + 1}. [${toTimestamp(
        Math.round(segment.t0),
      )} --> ${toTimestamp(Math.round(segment.t1))}] Duration: ${(
        (segment.t1 - segment.t0) / 100
      ).toFixed(2)}s`,
  )
  .join('\n')}`

                setVadResult(resultText)
                log(`Detected ${segments.length} speech segments (sensitive)`)
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
        {vadResult && (
          <View style={styles.logContainer}>
            <Text style={styles.logText}>{vadResult}</Text>
          </View>
        )}
        <Button
          title="Release VAD Context"
          style={styles.buttonClear}
          onPress={async () => {
            if (!vadContext) return
            await vadContext.release()
            vadContextRef.current = null
            log('Released VAD context')
          }}
        />
        <Button
          title="Clear Logs"
          style={styles.buttonClear}
          onPress={() => {
            setLogs([
              `whisper.cpp version: ${libVersion}`,
              'VAD Example - Voice Activity Detection',
            ])
            setVadResult('')
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
          title="Play Sample Audio"
          style={styles.buttonClear}
          onPress={async () => {
            const player = new Sound(sampleFile, '', (e) => {
              if (e) {
                log('error', e)
                return
              }
              log('Playing sample audio for VAD analysis...')
              player.play((success) => {
                if (success) {
                  log('Sample audio playback finished')
                } else {
                  log('playback failed due to audio decoding errors')
                }
                player.release()
              })
            })
          }}
        />
        <Button
          title="Play Recorded Audio"
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
              log('Playing recorded audio...')
              player.play((success) => {
                if (success) {
                  log('Recorded audio playback finished')
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
