import React, { useCallback, useEffect, useRef, useState } from 'react'
import { ScrollView, StyleSheet, Text, View } from 'react-native'
import RNFS from 'react-native-fs'
import { initParakeet } from '../../src'
import type { ParakeetContext } from '../../src'
import { Button } from './Button'
import {
  downloadParakeetModel,
  fileDir,
  parakeetModels,
  ParakeetModel,
  toTimestamp,
} from './util'

const sampleFile = require('../assets/jfk.wav')

const styles = StyleSheet.create({
  scrollview: { flexGrow: 1, justifyContent: 'center' },
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 4,
  },
  buttons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    margin: 8,
  },
  buttonClear: { backgroundColor: '#888' },
  configTitle: { fontSize: 16, fontWeight: 'bold', textAlign: 'center' },
  hint: { fontSize: 12, color: '#555', marginHorizontal: 12, textAlign: 'center' },
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

export default function ParakeetExample() {
  const contextRef = useRef<ParakeetContext | null>(null)
  const isMountedRef = useRef(true)
  const lastDownloadPercentRef = useRef(-1)
  const [logs, setLogs] = useState<string[]>([])
  const [transcribeResult, setTranscribeResult] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<ParakeetModel>('q4_0')
  const [downloadProgress, setDownloadProgress] = useState(0)
  const [isPreparing, setIsPreparing] = useState(false)
  const [stopTranscribe, setStopTranscribe] = useState<
    (() => Promise<void>) | null
  >(null)

  const log = useCallback((...messages: unknown[]) => {
    if (!isMountedRef.current) return
    setLogs((previous) => [...previous, messages.map(String).join(' ')])
  }, [])

  useEffect(
    () => () => {
      isMountedRef.current = false
      const context = contextRef.current
      contextRef.current = null
      void context?.release().catch(() => {})
    },
    [],
  )

  const downloadAndInitialize = async () => {
    setIsPreparing(true)
    setDownloadProgress(0)
    lastDownloadPercentRef.current = -1

    try {
      if (contextRef.current) {
        await contextRef.current.release()
        contextRef.current = null
        log('Released previous Parakeet context')
      }

      const modelPath = await downloadParakeetModel(
        selectedModel,
        (progress) => {
          if (!isMountedRef.current) return
          const percent = Math.round(progress * 100)
          setDownloadProgress(progress)
          if (percent > lastDownloadPercentRef.current) {
            lastDownloadPercentRef.current = percent
            log(`Download progress: ${percent}%`)
          }
        },
        (message) => log(message),
      )

      if (!isMountedRef.current) return

      log('Initialize Parakeet context...')
      const startedAt = Date.now()
      const context = await initParakeet({ filePath: modelPath })
      if (!isMountedRef.current) {
        await context.release()
        return
      }
      contextRef.current = context
      log('Loaded model, ID:', context.id)
      log(`Loaded model in ${Date.now() - startedAt}ms in ${mode} mode`)
      log(
        context.gpu
          ? 'GPU acceleration enabled'
          : `Using CPU${context.reasonNoGPU ? `: ${context.reasonNoGPU}` : ''}`,
      )
    } catch (error) {
      log('Error downloading or initializing model:', error)
    } finally {
      if (isMountedRef.current) {
        setDownloadProgress(0)
        setIsPreparing(false)
      }
    }
  }

  const transcribe = async () => {
    const context = contextRef.current
    if (!context) {
      log('No Parakeet context')
      return
    }

    log('Start transcribing bundled jfk.wav...')
    const startedAt = Date.now()
    const { stop, promise } = context.transcribe(sampleFile)
    setStopTranscribe(() => stop)

    try {
      const { result, segments, isAborted } = await promise
      const elapsed = Date.now() - startedAt
      setTranscribeResult(
        `${isAborted ? 'Transcription aborted' : `Transcribed result: ${result}`}` +
          `\nTranscribed in ${elapsed}ms in ${mode} mode` +
          `\nSegments:\n${segments
            .map(
              (segment) =>
                `[${toTimestamp(segment.t0)} --> ${toTimestamp(segment.t1)}]  ${segment.text}`,
            )
            .join('\n')}`,
      )
      log(isAborted ? 'Transcription aborted' : 'Finished transcribing')
    } catch (error) {
      log('Error transcribing:', error)
      setTranscribeResult('Error transcribing')
    } finally {
      setStopTranscribe(null)
    }
  }

  return (
    <ScrollView
      contentInsetAdjustmentBehavior="automatic"
      contentContainerStyle={styles.scrollview}
    >
      <View style={styles.container}>
        <Text style={styles.configTitle}>NVIDIA Parakeet TDT Demo</Text>
        <Text style={styles.hint}>
          Models are downloaded at runtime. q4_0 is the smallest option (about
          356 MB).
        </Text>

        <Text style={styles.configTitle}>Model Quantization</Text>
        <View style={styles.buttons}>
          {parakeetModels.map((model) => (
            <Button
              key={model}
              title={model}
              disabled={isPreparing || !!stopTranscribe}
              style={
                selectedModel === model ? { backgroundColor: '#007AFF' } : null
              }
              onPress={() => setSelectedModel(model)}
            />
          ))}
        </View>

        <Button
          title={
            isPreparing
              ? 'Downloading / Initializing...'
              : `Download & Initialize ${selectedModel}`
          }
          disabled={isPreparing || !!stopTranscribe}
          onPress={downloadAndInitialize}
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
            title="Transcribe jfk.wav"
            disabled={isPreparing || !!stopTranscribe}
            onPress={transcribe}
          />
          {stopTranscribe && (
            <Button
              title="Stop"
              onPress={() => {
                void stopTranscribe()
              }}
            />
          )}
        </View>

        {logs.length > 0 && (
          <View style={styles.logContainer}>
            {logs.map((message, index) => (
              <Text key={index} style={styles.logText}>
                {message}
              </Text>
            ))}
          </View>
        )}

        {transcribeResult && (
          <View style={styles.logContainer}>
            <Text style={styles.logText}>{transcribeResult}</Text>
          </View>
        )}

        <View style={styles.buttons}>
          <Button
            title="Release Context"
            style={styles.buttonClear}
            onPress={async () => {
              if (!contextRef.current) return
              await contextRef.current.release()
              contextRef.current = null
              log('Released Parakeet context')
            }}
          />
          <Button
            title="Clear Logs"
            style={styles.buttonClear}
            onPress={() => {
              setLogs([])
              setTranscribeResult(null)
            }}
          />
          <Button
            title="Clear Download Files"
            style={styles.buttonClear}
            onPress={async () => {
              await RNFS.unlink(fileDir).catch(() => {})
              log('Deleted downloaded files')
            }}
          />
        </View>
      </View>
    </ScrollView>
  )
}
