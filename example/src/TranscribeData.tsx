import React, { useCallback, useEffect, useRef, useState } from 'react'
import { StyleSheet, ScrollView, View, Text } from 'react-native'
import LiveAudioStream from '@fugood/react-native-audio-pcm-stream'
import { Buffer } from 'buffer'
import RNFS from 'react-native-fs'
import Sound from 'react-native-sound'
import { initWhisper, libVersion } from '../../src'
import type { WhisperContext } from '../../src'
import { Button } from './Button'
import contextOpts from './context-opts'
import { WavFileWriter } from './utils/WavFileWriter'
import { createDir, fileDir } from './utils/common'

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
const recordFile = `${fileDir}/record.wav`

const audioOptions = {
  sampleRate: 16000,
  channels: 1,
  bitsPerSample: 16,
  audioSource: 6,
  wavFile: recordFile,
  bufferSize: 16 * 1024,
}

export default function TranscribeData() {
  const whisperContextRef = useRef<WhisperContext | null>(null)
  const whisperContext = whisperContextRef.current
  const [logs, setLogs] = useState([`whisper.cpp version: ${libVersion}`])
  const [transcibeResult, setTranscibeResult] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const recordedDataRef = useRef<Uint8Array | null>(null)

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
      log('Started recording...')
    } catch (error) {
      log('Error starting recording:', error)
    }
  }

  const stopRecording = async () => {
    try {
      // Stop recording and get the wav file path
      await LiveAudioStream.stop()
      setIsRecording(false)
      log('Stopped recording')

      if (!recordedDataRef.current) return log('No recorded data')
      if (!whisperContext) return log('No context')

      const wavFileWriter = new WavFileWriter(recordFile, audioOptions)
      await wavFileWriter.initialize()
      await wavFileWriter.appendAudioData(Buffer.from(recordedDataRef.current!))
      await wavFileWriter.finalize()

      // Read the wav file as base64
      const base64Data = Buffer.from(recordedDataRef.current!).toString(
        'base64',
      )
      log('Start transcribing...')

      const startTime = Date.now()
      const { promise } = await whisperContext.transcribeData(base64Data, {
        language: 'en',
        onProgress: (progress) => {
          log(`Transcribing progress: ${progress}%`)
        },
      })
      const { result } = await promise
      const endTime = Date.now()

      setTranscibeResult(
        `Transcribed result: ${result}\n` +
          `Transcribed in ${endTime - startTime}ms in ${mode} mode`,
      )
      log('Finished transcribing')
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
            title="Initialize Context"
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

        <View style={styles.buttons}>
          <Button
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
            onPress={isRecording ? stopRecording : startRecording}
            disabled={!whisperContext}
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
            setTranscibeResult(null)
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
