/* eslint-disable react/jsx-one-expression-per-line */
/* eslint-disable no-use-before-define */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  StyleSheet,
  ScrollView,
  View,
  Text,
  Platform,
  PermissionsAndroid,
  Alert,
} from 'react-native'
import RNFS from 'react-native-fs'
import { initWhisper, initWhisperVad, libVersion } from '../../src'
import type { WhisperContext, WhisperVadContext } from '../../src'
import { Button } from './Button'
import contextOpts from './context-opts'
import { createDir, fileDir, toTimestamp } from './utils/common'
import { RealtimeTranscriber } from './realtime-transcription/RealtimeTranscriber'
import { LiveAudioStreamAdapter } from './realtime-transcription/LiveAudioStreamAdapter'
import type {
  TranscribeEvent,
  VADEvent,
  RealtimeOptions,
  StatsEvent,
  RealtimeTranscriberDependencies,
} from './realtime-transcription/types'
import { VAD_PRESETS } from './realtime-transcription/types'

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
  buttons: { flexDirection: 'row', flexWrap: 'wrap' },
  button: { margin: 4, backgroundColor: '#333', borderRadius: 4, padding: 8 },
  buttonClear: { backgroundColor: '#888' },
  buttonActive: { backgroundColor: '#4CAF50' },
  buttonText: { fontSize: 14, color: 'white', textAlign: 'center' },
  logContainer: {
    backgroundColor: 'lightgray',
    padding: 8,
    width: '95%',
    borderRadius: 8,
    marginVertical: 8,
  },
  logText: { fontSize: 12, color: '#333' },
  configContainer: {
    backgroundColor: '#f0f0f0',
    padding: 12,
    width: '95%',
    borderRadius: 8,
    marginVertical: 4,
  },
  configTitle: { fontSize: 16, fontWeight: 'bold', marginBottom: 8 },
  configRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 4,
  },
  configLabel: { fontSize: 14, color: '#666' },
  configValue: { fontSize: 14, fontWeight: '500' },
  statusContainer: {
    backgroundColor: '#e8f5e8',
    padding: 8,
    width: '95%',
    borderRadius: 8,
    marginVertical: 4,
  },
  statusActive: { backgroundColor: '#e8f5e8' },
  statusInactive: { backgroundColor: '#f5e8e8' },
  statusText: { fontSize: 12, color: '#333' },
})

const mode = process.env.NODE_ENV === 'development' ? 'debug' : 'release'

export default function RealtimeTranscriberDemo() {
  const whisperContextRef = useRef<WhisperContext | null>(null)
  const vadContextRef = useRef<WhisperVadContext | null>(null)
  const realtimeTranscriberRef = useRef<RealtimeTranscriber | null>(null)

  const [logs, setLogs] = useState([
    `Realtime Transcriber Demo - whisper.cpp v${libVersion}`,
  ])
  const [transcribeResult, setTranscribeResult] = useState<string | null>(null)
  const [currentVadPreset, setCurrentVadPreset] =
    useState<keyof typeof VAD_PRESETS>('DEFAULT')
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [realtimeStats, setRealtimeStats] = useState<any>(null)
  const [vadEvents, setVadEvents] = useState<VADEvent[]>([])

  const log = useCallback((...messages: any[]) => {
    setLogs((prev) => [
      ...prev,
      `${new Date().toLocaleTimeString()}: ${messages.join(' ')}`,
    ])
  }, [])

  useEffect(
    () => () => {
      // Cleanup on unmount
      whisperContextRef.current?.release()
      vadContextRef.current?.release()
      realtimeTranscriberRef.current?.release()
    },
    [],
  )

  const initializeContexts = async () => {
    try {
      if (whisperContextRef.current) {
        log('Found previous Whisper context')
        await whisperContextRef.current.release()
        whisperContextRef.current = null
        log('Released previous Whisper context')
      }

      if (vadContextRef.current) {
        log('Found previous VAD context')
        await vadContextRef.current.release()
        vadContextRef.current = null
        log('Released previous VAD context')
      }

      log('Initializing Whisper context...')
      const startTime = Date.now()
      const whisperCtx = await initWhisper({
        filePath: require('../assets/ggml-base.bin'),
        ...contextOpts,
      })
      const endTime = Date.now()
      log('Loaded Whisper model, ID:', whisperCtx.id)
      log('Loaded Whisper model in', endTime - startTime, `ms in ${mode} mode`)
      whisperContextRef.current = whisperCtx

      log('Initializing VAD context...')
      const vadStartTime = Date.now()
      const vadCtx = await initWhisperVad({
        filePath: require('../assets/ggml-silero-v5.1.2.bin'),
        useGpu: true,
        nThreads: 4,
      })
      const vadEndTime = Date.now()
      log('Loaded VAD model, ID:', vadCtx.id)
      log(
        'Loaded VAD model in',
        vadEndTime - vadStartTime,
        `ms in ${mode} mode`,
      )
      vadContextRef.current = vadCtx

      log('Both contexts initialized successfully!')
    } catch (error) {
      log('Error initializing contexts:', error)
      Alert.alert('Error', `Failed to initialize: ${error}`)
    }
  }

  const startRealtimeTranscription = async () => {
    if (!whisperContextRef.current || !vadContextRef.current) {
      Alert.alert('Error', 'Contexts not initialized')
      return
    }

    try {
      await createDir(log)

      if (!realtimeTranscriberRef.current) {
        const options: RealtimeOptions = {
          audioSliceSec: 30,
          audioMinSec: 0.5,
          maxSlicesInMemory: 1,
          vadPreset: currentVadPreset,
          vadOptions: VAD_PRESETS[currentVadPreset],
          transcribeOptions: {
            language: 'en',
            maxLen: 1,
          },
          audioOutputPath: `${fileDir}/realtime-recording.wav`,
          audioStreamConfig: {
            sampleRate: 16000,
            channels: 1,
            bitsPerSample: 16,
            bufferSize: 16 * 1024,
            audioSource: 6,
          },
        }

        // Create audio stream adapter
        const audioStream = new LiveAudioStreamAdapter()
        await audioStream.initialize(options.audioStreamConfig!)

        // Create dependencies
        const dependencies: RealtimeTranscriberDependencies = {
          contexts: {
            whisperContext: whisperContextRef.current,
            vadContext: vadContextRef.current,
          },
          audioStream,
        }

        // Create RealtimeTranscriber if not exists
        const transcriber = new RealtimeTranscriber(dependencies, options, {
          onTranscribe: handleTranscribeEvent,
          onVAD: handleVADEvent,
          onError: handleError,
          onStatusChange: handleStatusChange,
          onStatsUpdate: handleStatsUpdate,
        })

        realtimeTranscriberRef.current = transcriber
      }
      // Start transcription
      await realtimeTranscriberRef.current.start()
      log('Realtime transcription started')
    } catch (error) {
      log('Error starting realtime transcription:', error)
      Alert.alert('Error', `Failed to start: ${error}`)
    }
  }

  const stopRealtimeTranscription = async () => {
    if (!realtimeTranscriberRef.current) {
      return
    }

    try {
      await realtimeTranscriberRef.current.stop()
      setRealtimeStats(null)
      log('Realtime transcription stopped')
    } catch (error) {
      log('Error stopping realtime transcription:', error)
    }
  }

  const handleTranscribeEvent = (event: TranscribeEvent) => {
    const { data, sliceIndex } = event

    if (data?.result) {
      // Get all transcription results from the transcriber
      const allResults =
        realtimeTranscriberRef.current?.getTranscriptionResults() || []

      if (allResults.length > 0) {
        const separator = `\n\n${'='.repeat(50)}\n\n`
        const formattedResults = allResults
          .map(({ slice, transcribeEvent }) => {
            const { data: resultData, processTime: procTime } = transcribeEvent
            if (!resultData) return null

            return (
              `[Slice ${slice.index}] ${resultData.result}\n` +
              `Process Time: ${procTime}ms | Duration: ${(
                (slice.endTime - slice.startTime) /
                1000
              ).toFixed(1)}s\n` +
              `Memory: ${
                transcribeEvent.memoryUsage?.slicesInMemory || 0
              } slices, ${transcribeEvent.memoryUsage?.estimatedMB || 0}MB\n` +
              `Segments:\n${resultData.segments
                .map(
                  (segment) =>
                    `  [${toTimestamp(segment.t0)} --> ${toTimestamp(
                      segment.t1,
                    )}] ${segment.text}`,
                )
                .join('\n')}`
            )
          })
          .filter((result): result is string => result !== null)
          .join(separator)

        setTranscribeResult(formattedResults)
      }

      log(
        `Transcribed slice ${sliceIndex}: "${data.result.substring(
          0,
          50,
        )}..." (Total results: ${allResults.length})`,
      )
    }
  }

  const handleVADEvent = (vadEvent: VADEvent) => {
    setVadEvents((prev) => [...prev.slice(-19), vadEvent]) // Keep last 20 events

    if (vadEvent.type !== 'silence') {
      log(
        `VAD: ${vadEvent.type} (confidence: ${vadEvent.confidence.toFixed(2)})`,
      )
    }
  }

  const handleStatsUpdate = (statsEvent: StatsEvent) => {
    setRealtimeStats(statsEvent.data)

    // Log significant changes
    if (statsEvent.type === 'status_change') {
      log(
        `Status changed: ${
          statsEvent.data.isActive ? 'ACTIVE' : 'INACTIVE'
        }, transcribing: ${statsEvent.data.isTranscribing}`,
      )
    } else if (statsEvent.type === 'queue_change') {
      log(`Queue length: ${statsEvent.data.queueLength}`)
    } else if (statsEvent.type === 'memory_change') {
      const memMB = statsEvent.data.sliceStats?.memoryUsage?.estimatedMB || 0
      log(`Memory usage: ${memMB.toFixed(1)}MB`)
    }
  }

  const handleError = (error: string) => {
    log('Realtime Error:', error)
  }

  const handleStatusChange = (isActive: boolean) => {
    setIsTranscribing(isActive)
    log(`Realtime status: ${isActive ? 'ACTIVE' : 'INACTIVE'}`)
  }

  const changeVadPreset = () => {
    const presetKeys = Object.keys(VAD_PRESETS) as Array<
      keyof typeof VAD_PRESETS
    >
    const currentIndex = presetKeys.indexOf(currentVadPreset)
    const nextIndex = (currentIndex + 1) % presetKeys.length
    const nextPreset = presetKeys[nextIndex] as keyof typeof VAD_PRESETS

    setCurrentVadPreset(nextPreset)
    log(`VAD preset changed to: ${nextPreset}`)

    // Update transcriber if active
    if (realtimeTranscriberRef.current) {
      realtimeTranscriberRef.current.updateVadOptions(VAD_PRESETS[nextPreset])
    }
  }

  const resetAll = () => {
    if (realtimeTranscriberRef.current) {
      realtimeTranscriberRef.current.reset()
    }
    setTranscribeResult(null)
    setVadEvents([])
    setRealtimeStats(null)
    log('Reset all components')
  }

  const checkRecordedFile = async () => {
    const recordFilePath = `${fileDir}/realtime-recording.wav`

    try {
      const exists = await RNFS.exists(recordFilePath)
      if (!exists) {
        Alert.alert(
          'Info',
          'No recorded file found. Start a realtime session first.',
        )
        return
      }

      const stats = await RNFS.stat(recordFilePath)
      const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2)

      Alert.alert(
        'Recorded File Info',
        `File: realtime-recording.wav\nSize: ${fileSizeMB} MB\nPath: ${recordFilePath}`,
        [
          { text: 'OK', style: 'default' },
          {
            text: 'Delete File',
            style: 'destructive',
            onPress: deleteRecordedFile,
          },
        ],
      )

      log(`Found recorded file: ${fileSizeMB} MB`)
    } catch (error) {
      log('Error checking recorded file:', error)
      Alert.alert('Error', `Failed to check recorded file: ${error}`)
    }
  }

  const deleteRecordedFile = async () => {
    const recordFilePath = `${fileDir}/realtime-recording.wav`

    try {
      const exists = await RNFS.exists(recordFilePath)
      if (exists) {
        await RNFS.unlink(recordFilePath)
        log('Deleted recorded file')
        Alert.alert('Success', 'Recorded file deleted')
      }
    } catch (error) {
      log('Error deleting recorded file:', error)
      Alert.alert('Error', `Failed to delete file: ${error}`)
    }
  }

  const forceNextSlice = async () => {
    if (!realtimeTranscriberRef.current) {
      Alert.alert('Error', 'Realtime transcriber not initialized')
      return
    }

    try {
      log('Forcing next slice...')
      await realtimeTranscriberRef.current.nextSlice()
      log('Successfully forced next slice')
    } catch (error) {
      log('Error forcing next slice:', error)
      Alert.alert('Error', `Failed to force next slice: ${error}`)
    }
  }

  return (
    <ScrollView
      contentInsetAdjustmentBehavior="automatic"
      contentContainerStyle={styles.scrollview}
    >
      <View style={styles.container}>
        {/* Initialization */}
        <View style={styles.buttons}>
          <Button title="Initialize Contexts" onPress={initializeContexts} />
        </View>

        {/* VAD Configuration */}
        <View style={styles.configContainer}>
          <Text style={styles.configTitle}>VAD Configuration</Text>
          <View style={styles.configRow}>
            <Text style={styles.configLabel}>Current Preset:</Text>
            <Text style={styles.configValue}>{currentVadPreset}</Text>
          </View>
          <Button
            title="Change VAD Preset"
            onPress={changeVadPreset}
            style={styles.buttonClear}
          />
        </View>

        {/* Realtime Controls */}
        <View style={styles.buttons}>
          <Button
            title={isTranscribing ? 'Stop Realtime' : 'Start Realtime'}
            onPress={
              isTranscribing
                ? stopRealtimeTranscription
                : startRealtimeTranscription
            }
            style={isTranscribing ? styles.buttonActive : undefined}
            disabled={!whisperContextRef.current}
          />
          <Button
            title="Force Next Slice"
            onPress={forceNextSlice}
            style={styles.buttonClear}
            disabled={!isTranscribing || !realtimeTranscriberRef.current}
          />
          <Button
            title="Reset All"
            onPress={resetAll}
            style={styles.buttonClear}
          />
        </View>

        {/* Status Display */}
        {realtimeStats && (
          <View
            style={[
              styles.statusContainer,
              isTranscribing ? styles.statusActive : styles.statusInactive,
            ]}
          >
            <Text style={styles.statusText}>
              Status: {isTranscribing ? 'TRANSCRIBING' : 'STOPPED'} | VAD:{' '}
              {realtimeStats.vadEnabled ? 'ON' : 'OFF'} | Queue:{' '}
              {realtimeStats.queueLength} | Memory:{' '}
              {realtimeStats.sliceStats?.memoryUsage?.estimatedMB || 0}MB
            </Text>
            <Text style={styles.statusText}>
              Slices: {realtimeStats.sliceStats?.currentSliceIndex || 0}{' '}
              current,
              {realtimeStats.sliceStats?.transcribeSliceIndex || 0} transcribing
              | VAD Threshold:{' '}
              {realtimeStats.vadStats?.currentThreshold?.toFixed(2) || 'N/A'}
            </Text>
          </View>
        )}

        {/* VAD Events Display */}
        {vadEvents.length > 0 && (
          <View style={styles.logContainer}>
            <Text style={styles.configTitle}>Recent VAD Events</Text>
            {vadEvents.slice(-5).map((event, index) => (
              <Text key={index} style={styles.logText}>
                {event.type}: {event.confidence.toFixed(2)} confidence (slice{' '}
                {event.sliceIndex})
              </Text>
            ))}
          </View>
        )}

        {/* Transcription Result */}
        {transcribeResult && (
          <View style={styles.logContainer}>
            <Text style={styles.configTitle}>Latest Transcription</Text>
            <Text style={styles.logText}>{transcribeResult}</Text>
          </View>
        )}

        {/* Logs */}
        <View style={styles.logContainer}>
          <Text style={styles.configTitle}>Debug Logs</Text>
          {logs.slice(-10).map((msg, index) => (
            <Text key={index} style={styles.logText}>
              {msg}
            </Text>
          ))}
        </View>

        {/* Cleanup */}
        <View style={styles.buttons}>
          <Button
            title="Clear Logs"
            style={styles.buttonClear}
            onPress={() => {
              setLogs([
                `Realtime Transcriber Demo - whisper.cpp v${libVersion}`,
              ])
              setTranscribeResult(null)
              setVadEvents([])
            }}
          />
          <Button
            title="Check Recorded File"
            style={styles.buttonClear}
            onPress={checkRecordedFile}
          />
        </View>
      </View>
    </ScrollView>
  )
}
