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
  Switch,
} from 'react-native'
import RNFS from 'react-native-fs'
import { initWhisper, initWhisperVad, libVersion } from '../../src'
import type { WhisperContext, WhisperVadContext } from '../../src'
import { Button } from './Button'
import contextOpts from './context-opts'
import { createDir, fileDir, toTimestamp } from './utils/common'
import {
  RealtimeTranscriber,
  LiveAudioStreamAdapter,
  SimulateFileAudioStreamAdapter,
  VAD_PRESETS,
  type TranscribeEvent,
  type VADEvent,
  type RealtimeOptions,
  type StatsEvent,
  type RealtimeTranscriberDependencies,
  type AudioStreamInterface,
} from './realtime-transcription'

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
  buttonDanger: { backgroundColor: '#f44336' },
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
    alignItems: 'center',
    marginVertical: 4,
  },
  configLabel: { fontSize: 14, color: '#666' },
  configValue: { fontSize: 14, fontWeight: '500' },
  playbackContainer: {
    backgroundColor: '#fff3cd',
    padding: 12,
    width: '95%',
    borderRadius: 8,
    marginVertical: 4,
  },
  playbackCompleted: {
    backgroundColor: '#d4edda',
  },
  playbackTitle: { fontSize: 16, fontWeight: 'bold', marginBottom: 8 },
  progressBar: {
    height: 4,
    backgroundColor: '#ddd',
    borderRadius: 2,
    marginVertical: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007bff',
    borderRadius: 2,
  },
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

// JFK audio file URL from whisper.cpp repository
const JFK_AUDIO_URL =
  'https://github.com/ggml-org/whisper.cpp/raw/refs/heads/master/samples/jfk.wav'

export default function RealtimeTranscriberDemo() {
  const whisperContextRef = useRef<WhisperContext | null>(null)
  const vadContextRef = useRef<WhisperVadContext | null>(null)
  const realtimeTranscriberRef = useRef<RealtimeTranscriber | null>(null)
  const audioStreamRef = useRef<AudioStreamInterface | null>(null)

  const [logs, setLogs] = useState([
    `Realtime Transcriber Demo - whisper.cpp v${libVersion}`,
  ])
  const [transcribeResult, setTranscribeResult] = useState<string | null>(null)
  const [currentVadPreset, setCurrentVadPreset] =
    useState<keyof typeof VAD_PRESETS>('DEFAULT')
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [realtimeStats, setRealtimeStats] = useState<any>(null)
  const [vadEvents, setVadEvents] = useState<VADEvent[]>([])

  // Auto-slice configuration
  const [autoSliceOnSpeechEnd, setAutoSliceOnSpeechEnd] = useState(false)
  const autoSliceThreshold = 0.85 // Fixed 85% threshold

  // File simulation specific state
  const [useFileSimulation, setUseFileSimulation] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const [simulationStats, setSimulationStats] = useState<any>(null)
  const [audioFilePath, setAudioFilePath] = useState<string | null>(null)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState(0)

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
      audioStreamRef.current?.release()
    },
    [],
  )

  // Update simulation stats periodically when using file simulation
  useEffect(() => {
    if (!useFileSimulation || !audioStreamRef.current) {
      return undefined
    }

    const interval = setInterval(() => {
      if (audioStreamRef.current && 'getStatistics' in audioStreamRef.current) {
        const stats = (audioStreamRef.current as any).getStatistics()
        setSimulationStats(stats)
      }
    }, 500) // Update every 500ms

    return () => clearInterval(interval)
  }, [useFileSimulation, isTranscribing])

  const downloadAudioFile = async () => {
    if (audioFilePath) {
      // File already downloaded
      return audioFilePath
    }

    setIsDownloading(true)
    setDownloadProgress(0)

    try {
      const downloadPath = `${fileDir}/jfk-sample.wav`

      // Check if file already exists
      const exists = await RNFS.exists(downloadPath)
      if (exists) {
        log('Audio file already exists, using cached version')
        setAudioFilePath(downloadPath)
        setIsDownloading(false)
        return downloadPath
      }

      log('Downloading JFK audio sample from whisper.cpp repository...')

      const downloadResult = await RNFS.downloadFile({
        fromUrl: JFK_AUDIO_URL,
        toFile: downloadPath,
        progress: (res) => {
          const progress = (res.bytesWritten / res.contentLength) * 100
          setDownloadProgress(progress)
          log(`Download progress: ${progress.toFixed(1)}%`)
        },
      }).promise

      if (downloadResult.statusCode === 200) {
        log('Audio file downloaded successfully')
        setAudioFilePath(downloadPath)
        setIsDownloading(false)
        setDownloadProgress(100)
        return downloadPath
      } else {
        throw new Error(
          `Download failed with status: ${downloadResult.statusCode}`,
        )
      }
    } catch (error) {
      log('Error downloading audio file:', error)
      setIsDownloading(false)
      setDownloadProgress(0)
      Alert.alert('Download Error', `Failed to download audio file: ${error}`)
      throw error
    }
  }

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
          autoSliceOnSpeechEnd,
          autoSliceThreshold,
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

        // Create appropriate audio stream adapter
        let audioStream: AudioStreamInterface

        if (useFileSimulation) {
          log('Creating file simulation adapter...')

          // Download audio file if needed
          try {
            const filePath = await downloadAudioFile()

            audioStream = new SimulateFileAudioStreamAdapter({
              filePath,
              playbackSpeed,
              chunkDurationMs: 100,
              loop: false,
              onEndOfFile: () => {
                log('File simulation reached end - no new buffer available')
                log('Automatically stopping realtime transcription...')

                // Automatically stop realtime transcription when file ends
                setTimeout(() => {
                  stopRealtimeTranscription()
                }, 1000) // Small delay to allow final processing
              },
            })
          } catch (error) {
            log('Failed to download audio file for simulation')
            Alert.alert(
              'Error',
              'Could not download audio file for simulation. Please check your internet connection and try again.',
            )
            return
          }
        } else {
          log('Creating live audio adapter...')
          audioStream = new LiveAudioStreamAdapter()
        }

        await audioStream.initialize(options.audioStreamConfig!)
        audioStreamRef.current = audioStream

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

      if (useFileSimulation && audioStreamRef.current) {
        ;(audioStreamRef.current as any).resetBuffer()
      }

      // Start transcription
      await realtimeTranscriberRef.current.start()
      log(
        `Realtime transcription started (${
          useFileSimulation ? 'File Simulation - JFK Speech' : 'Live Audio'
        })`,
      )
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
      setSimulationStats(null)
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

  const changePlaybackSpeed = () => {
    const speeds = [0.5, 1.0, 1.5, 2.0]
    const currentIndex = speeds.indexOf(playbackSpeed)
    const nextIndex = (currentIndex + 1) % speeds.length
    const nextSpeed = speeds[nextIndex] || 1.0

    setPlaybackSpeed(nextSpeed)
    log(`Playback speed changed to: ${nextSpeed}x`)

    // Update adapter if active and using file simulation
    if (
      audioStreamRef.current &&
      'setPlaybackSpeed' in audioStreamRef.current
    ) {
      ;(audioStreamRef.current as any).setPlaybackSpeed(nextSpeed)
    }
  }

  const seekToPosition = (percentage: number) => {
    if (
      !useFileSimulation ||
      !audioStreamRef.current ||
      !('seekToTime' in audioStreamRef.current) ||
      !simulationStats
    ) {
      return
    }

    const targetTime = simulationStats.totalDuration * (percentage / 100)
    ;(audioStreamRef.current as any).seekToTime(targetTime)
    log(`Seeked to ${targetTime.toFixed(1)}s (${percentage}%)`)
  }

  const resetAll = () => {
    if (realtimeTranscriberRef.current) {
      realtimeTranscriberRef.current.reset()
    }
    setTranscribeResult(null)
    setVadEvents([])
    setRealtimeStats(null)
    setSimulationStats(null)
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

        {/* Audio Source Configuration */}
        <View style={styles.configContainer}>
          <Text style={styles.configTitle}>Audio Source</Text>
          {useFileSimulation && (
            <Text style={styles.configLabel}>
              Using JFK speech sample from whisper.cpp repository
            </Text>
          )}
          <View style={styles.configRow}>
            <Text style={styles.configLabel}>Use File Simulation:</Text>
            <Switch
              value={useFileSimulation}
              onValueChange={(value) => {
                setUseFileSimulation(value)
                log(`Audio source: ${value ? 'File Simulation' : 'Live Audio'}`)
              }}
              disabled={isTranscribing}
            />
          </View>
          {useFileSimulation && (
            <>
              <View style={styles.configRow}>
                <Text style={styles.configLabel}>Audio File:</Text>
                <Text style={styles.configValue}>
                  {audioFilePath ? 'Downloaded' : 'Not downloaded'}
                </Text>
              </View>
              {isDownloading && (
                <View style={styles.configRow}>
                  <Text style={styles.configLabel}>Download Progress:</Text>
                  <Text style={styles.configValue}>
                    {downloadProgress.toFixed(1)}%
                  </Text>
                </View>
              )}
              <View style={styles.buttons}>
                <Button
                  title={audioFilePath ? 'Re-download Audio' : 'Download Audio'}
                  onPress={() => {
                    setAudioFilePath(null) // Force re-download
                    downloadAudioFile()
                  }}
                  style={styles.buttonClear}
                  disabled={isDownloading || isTranscribing}
                />
                {audioFilePath && (
                  <Button
                    title="Clear Cache"
                    onPress={async () => {
                      try {
                        if (audioFilePath) {
                          const exists = await RNFS.exists(audioFilePath)
                          if (exists) {
                            await RNFS.unlink(audioFilePath)
                            log('Audio file cache cleared')
                          }
                        }
                        setAudioFilePath(null)
                      } catch (error) {
                        log('Error clearing cache:', error)
                      }
                    }}
                    style={styles.buttonClear}
                    disabled={isDownloading || isTranscribing}
                  />
                )}
              </View>
              <View style={styles.configRow}>
                <Text style={styles.configLabel}>Playback Speed:</Text>
                <Text style={styles.configValue}>{playbackSpeed}x</Text>
              </View>
              <Button
                title="Change Speed"
                onPress={changePlaybackSpeed}
                style={styles.buttonClear}
                disabled={isTranscribing}
              />
            </>
          )}
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

        {/* Auto-Slice Configuration */}
        <View style={styles.configContainer}>
          <Text style={styles.configTitle}>Auto-Slice Configuration</Text>
          <Text style={styles.configLabel}>
            Automatically slice when speech ends and duration ≥{' '}
            {(autoSliceThreshold * 100).toFixed(0)}% of target
          </Text>
          <View style={styles.configRow}>
            <Text style={styles.configLabel}>Auto-Slice on Speech End:</Text>
            <Switch
              value={autoSliceOnSpeechEnd}
              onValueChange={(value) => {
                setAutoSliceOnSpeechEnd(value)
                log(
                  `Auto-slice on speech end: ${value ? 'ENABLED' : 'DISABLED'}`,
                )

                // Update transcriber if active
                if (realtimeTranscriberRef.current) {
                  realtimeTranscriberRef.current.updateAutoSliceOptions({
                    autoSliceOnSpeechEnd: value,
                  })
                }
              }}
              disabled={isTranscribing}
            />
          </View>
          <View style={styles.configRow}>
            <Text style={styles.configLabel}>Threshold:</Text>
            <Text style={styles.configValue}>
              {(autoSliceThreshold * 100).toFixed(0)}%
            </Text>
          </View>
          {autoSliceOnSpeechEnd && (
            <Text style={styles.configLabel}>
              Will auto-slice when speech ends and slice duration ≥{' '}
              {(30 * autoSliceThreshold).toFixed(1)}s
            </Text>
          )}
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
            style={isTranscribing ? styles.buttonDanger : styles.buttonActive}
            disabled={
              !whisperContextRef.current ||
              (useFileSimulation && !audioFilePath) ||
              isDownloading
            }
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

        {/* File Simulation Playback Controls */}
        {useFileSimulation && simulationStats && (
          <View
            style={[
              styles.playbackContainer,
              simulationStats.hasReachedEnd && styles.playbackCompleted,
            ]}
          >
            <Text style={styles.playbackTitle}>File Playback Progress</Text>
            <Text style={styles.configValue}>
              {simulationStats.currentTime.toFixed(1)}s /{' '}
              {simulationStats.totalDuration.toFixed(1)}s (
              {(simulationStats.progress * 100).toFixed(1)}%)
              {simulationStats.hasReachedEnd && ' - COMPLETED'}
            </Text>
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${simulationStats.progress * 100}%` },
                ]}
              />
            </View>
            <View style={styles.buttons}>
              <Button
                title="0%"
                onPress={() => seekToPosition(0)}
                style={styles.buttonClear}
                disabled={!isTranscribing || simulationStats.hasReachedEnd}
              />
              <Button
                title="25%"
                onPress={() => seekToPosition(25)}
                style={styles.buttonClear}
                disabled={!isTranscribing || simulationStats.hasReachedEnd}
              />
              <Button
                title="50%"
                onPress={() => seekToPosition(50)}
                style={styles.buttonClear}
                disabled={!isTranscribing || simulationStats.hasReachedEnd}
              />
              <Button
                title="75%"
                onPress={() => seekToPosition(75)}
                style={styles.buttonClear}
                disabled={!isTranscribing || simulationStats.hasReachedEnd}
              />
            </View>
          </View>
        )}

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
              current, {realtimeStats.sliceStats?.transcribeSliceIndex || 0}{' '}
              transcribing | Audio Source:{' '}
              {useFileSimulation ? 'File (JFK)' : 'Live'}
              {useFileSimulation && ` @ ${playbackSpeed}x`}
            </Text>
            <Text style={styles.statusText}>
              Auto-Slice:{' '}
              {realtimeStats.autoSliceConfig?.enabled ? 'ENABLED' : 'DISABLED'}
              {realtimeStats.autoSliceConfig?.enabled &&
                ` (≥${(realtimeStats.autoSliceConfig.threshold * 100).toFixed(
                  0,
                )}% = ${(
                  realtimeStats.autoSliceConfig.targetDuration *
                  realtimeStats.autoSliceConfig.threshold
                ).toFixed(1)}s)`}
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
