import React, { useState } from 'react'
import { View, Text, Button, StyleSheet, ScrollView } from 'react-native'
import RNFS from 'react-native-fs'
import {
  initWhisper,
  initWhisperVad,
  releaseAllWhisper,
  installJSIBindings,
  WhisperContext,
  WhisperVadContext,
} from '../../src'
import { WavFileReader } from './utils/WavFileReader'

// JFK audio file URL from whisper.cpp repository
const JFK_AUDIO_URL =
  'https://github.com/ggml-org/whisper.cpp/raw/refs/heads/master/samples/jfk.wav'

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  statusContainer: {
    backgroundColor: '#e0e0e0',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  statusText: {
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  statusSubText: {
    fontSize: 14,
    fontWeight: '400',
    textAlign: 'center',
    color: '#666',
    marginTop: 4,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  resultsContainer: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  resultText: {
    fontSize: 12,
    marginBottom: 5,
    fontFamily: 'Courier',
  },
})

const JSITest: React.FC = () => {
  const [testResults, setTestResults] = useState<string[]>([])
  const [whisperContext, setWhisperContext] = useState<WhisperContext | null>(null)
  const [vadContext, setVadContext] = useState<WhisperVadContext | null>(null)
  const [contextsInitialized, setContextsInitialized] = useState(false)

  const jsiAvailable =
    typeof (global as any).whisperTranscribeData === 'function' &&
    typeof (global as any).whisperVadDetectSpeech === 'function'

  const addTestResult = (result: string) => {
    setTestResults((prev) => [...prev, result])
  }

  const testJSIFunctions = async () => {
    // Download the JFK audio file
    const jfkAudioPath = `${RNFS.DocumentDirectoryPath}/jfk.wav`
    if (!(await RNFS.exists(jfkAudioPath))) {
      await RNFS.downloadFile({
        fromUrl: JFK_AUDIO_URL,
        toFile: jfkAudioPath,
      }).promise
    }

    const wavFileReader = new WavFileReader(jfkAudioPath)
    await wavFileReader.initialize()

    const audioData = wavFileReader.getAudioData()

    setTestResults([])

    if (!jsiAvailable) {
      addTestResult('❌ JSI functions not available')
      return
    }

    if (!contextsInitialized || !whisperContext) {
      addTestResult(
        '❌ Contexts not initialized. Please initialize contexts first.',
      )
      return
    }

    try {
      addTestResult('🧪 Converting audio data to ArrayBuffer...')

      // Convert audio data to 16-bit PCM ArrayBuffer
      const arrayBuffer = new ArrayBuffer(audioData!.length)
      const dataView = new DataView(arrayBuffer)
      for (let i = 0; i < audioData!.length; i += 1) {
        dataView.setUint8(i, audioData![i]!)
      }

      addTestResult(`✅ ArrayBuffer created: ${arrayBuffer.byteLength} bytes`)

      // Test 1: VAD detection with ArrayBuffer (if VAD context is available)
      if (vadContext) {
        addTestResult('🧪 Testing VAD detectSpeechData with ArrayBuffer...')

        try {
          const vadResult = await vadContext.detectSpeechData(arrayBuffer)
          addTestResult(
            `✅ VAD detection success: Found ${vadResult.length} speech segments`,
          )
          vadResult.forEach((segment, index) => {
            addTestResult(
              `  Segment ${index + 1}: ${segment.t0}ms - ${segment.t1}ms`,
            )
          })
        } catch (error) {
          addTestResult(`❌ VAD detection error: ${error}`)
        }
      } else {
        addTestResult(
          '⚠️ Skipping VAD test (VAD context not available)',
        )
      }

            // Test 2: Transcription with ArrayBuffer and callbacks
      addTestResult('🧪 Testing transcribeData with ArrayBuffer and callbacks...')

      try {
        let progressCount = 0
        let segmentsCount = 0

        const { promise: transcribePromise } = whisperContext.transcribeData(arrayBuffer, {
          language: 'en',
          maxThreads: 2,
          translate: false,
          tokenTimestamps: false,
          tdrzEnable: false,
          onProgress: (progress: number) => {
            progressCount += 1
            addTestResult(`📊 Progress callback #${progressCount}: ${progress}%`)
          },
          onNewSegments: (result: any) => {
            segmentsCount += 1
            addTestResult(`🆕 New segments callback #${segmentsCount}:`)
            addTestResult(`  New segments: ${result.nNew}`)
            addTestResult(`  Total segments: ${result.totalNNew}`)
            addTestResult(`  Text: "${result.result}"`)
          },
        })

        const transcribeResult = await transcribePromise
        addTestResult('✅ Transcription completed!')
        addTestResult(`📝 Final result: "${transcribeResult.result}"`)
        addTestResult(`📊 Total progress callbacks: ${progressCount}`)
        addTestResult(`🆕 Total segment callbacks: ${segmentsCount}`)
        addTestResult(`🔢 Final segments count: ${transcribeResult.segments?.length || 0}`)

      } catch (error) {
        addTestResult(`❌ Transcription error: ${error}`)
      }

    } catch (error) {
      addTestResult(`❌ Test error: ${error}`)
    }
  }

  const clearResults = () => {
    setTestResults([])
  }

  const initializeContexts = async () => {
    setTestResults([])
    addTestResult('🔄 Initializing contexts...')

    try {
      // Install JSI bindings first
      addTestResult('📦 Installing JSI bindings...')
      await installJSIBindings()
      addTestResult('✅ JSI bindings installed')

      // Initialize Whisper context
      addTestResult('🧠 Initializing Whisper context...')
      const newWhisperContext = await initWhisper({
        filePath: require('../assets/ggml-base.bin'),
      })
      setWhisperContext(newWhisperContext)
      addTestResult(
        `✅ Whisper context initialized with ID: ${newWhisperContext.id}`,
      )

      // Initialize VAD context (if available)
      try {
        addTestResult('🎤 Initializing VAD context...')
        const newVadContext = await initWhisperVad({
          filePath: require('../assets/ggml-silero-v5.1.2.bin'),
        })
        setVadContext(newVadContext)
        addTestResult(`✅ VAD context initialized with ID: ${newVadContext.id}`)
      } catch (vadError) {
        addTestResult(`⚠️ VAD context initialization failed: ${vadError}`)
        addTestResult('🧪 VAD tests will be skipped')
      }

      setContextsInitialized(true)
      addTestResult('🎉 All contexts initialized successfully!')
    } catch (error) {
      addTestResult(`❌ Context initialization failed: ${error}`)
      setContextsInitialized(false)
    }
  }

  const releaseContexts = async () => {
    try {
      addTestResult('🗑️ Releasing all contexts...')
      await releaseAllWhisper()
      setWhisperContext(null)
      setVadContext(null)
      setContextsInitialized(false)
      addTestResult('✅ All contexts released')
    } catch (error) {
      addTestResult(`❌ Error releasing contexts: ${error}`)
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>JSI Functions Test</Text>

      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>
          {`JSI Status: ${jsiAvailable ? '✅ Available' : '❌ Not Available'}`}
        </Text>
        <Text style={styles.statusText}>
          {`Contexts: ${
            contextsInitialized ? '✅ Initialized' : '❌ Not Initialized'
          }`}
        </Text>
        {whisperContext && (
          <Text style={styles.statusSubText}>
            {`Whisper Context ID: ${whisperContext.id}`}
          </Text>
        )}
        {vadContext && (
          <Text style={styles.statusSubText}>
            {`VAD Context ID: ${vadContext.id}`}
          </Text>
        )}
      </View>

      <View style={styles.buttonContainer}>
        <Button
          title="Initialize Contexts"
          onPress={initializeContexts}
          color={contextsInitialized ? '#4CAF50' : '#2196F3'}
        />
        <Button
          title="Test JSI Functions"
          onPress={testJSIFunctions}
          disabled={!contextsInitialized}
        />
      </View>

      <View style={styles.buttonContainer}>
        <Button title="Release Contexts" onPress={releaseContexts} />
        <Button title="Clear Results" onPress={clearResults} />
      </View>

      <ScrollView style={styles.resultsContainer}>
        <Text style={styles.resultsTitle}>Test Results:</Text>
        {testResults.map((result, index) => (
          <Text key={index} style={styles.resultText}>
            {result}
          </Text>
        ))}
      </ScrollView>
    </View>
  )
}

export default JSITest
