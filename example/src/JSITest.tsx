import React, { useState } from 'react'
import { View, Text, Button, StyleSheet, ScrollView } from 'react-native'
import RNFS from 'react-native-fs'
import {
  initWhisper,
  initWhisperVad,
  releaseAllWhisper,
  installJSIBindings,
} from '../../src'
import { WavFileReader } from './utils/WavFileReader'


// Declare the global JSI functions
declare global {
  // eslint-disable-next-line no-var
  var whisperTestContext: (
    contextId: number,
    arrayBuffer: ArrayBuffer,
  ) => boolean
  // eslint-disable-next-line no-var
  var whisperTranscribeData: (
    contextId: number,
    options: any,
    arrayBuffer: ArrayBuffer,
  ) => Promise<any>
  // eslint-disable-next-line no-var
  var whisperVadDetectSpeech: (
    contextId: number,
    options: any,
    arrayBuffer: ArrayBuffer,
  ) => Promise<any>
}

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
  const [contextId, setContextId] = useState<number | null>(null)
  const [vadContextId, setVadContextId] = useState<number | null>(null)
  const [contextsInitialized, setContextsInitialized] = useState(false)

  const jsiAvailable =
    typeof global.whisperTestContext === 'function' &&
    typeof global.whisperTranscribeData === 'function' &&
    typeof global.whisperVadDetectSpeech === 'function'

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

    if (!contextsInitialized || !contextId) {
      addTestResult(
        '❌ Contexts not initialized. Please initialize contexts first.',
      )
      return
    }

    try {
      // Test 1: whisperTestContext
      addTestResult('🧪 Testing whisperTestContext...')

      const arrayBuffer = new ArrayBuffer(audioData!.length)
      const dataView = new DataView(arrayBuffer)
      for (let i = 0; i < audioData!.length; i += 1) {
        dataView.setUint8(i, audioData![i]!)
      }

      const testResult: any = global.whisperTestContext(contextId, arrayBuffer)
      addTestResult(`✅ whisperTestContext result: ${testResult}`)

      // Test 2: whisperVadDetectSpeech (only if VAD context is available)
      if (vadContextId) {
        addTestResult('🧪 Testing whisperVadDetectSpeech...')

        const vadOptions = {
          threshold: 0.6,
          minSpeechDuration: 100,
          minSilenceDuration: 100,
        }

        global
          .whisperVadDetectSpeech(vadContextId, vadOptions, arrayBuffer)
          .then((result) => {
            addTestResult(
              `✅ whisperVadDetectSpeech success: ${JSON.stringify(
                result,
                null,
                2,
              )}`,
            )
          })
          .catch((error) => {
            addTestResult(
              `❌ whisperVadDetectSpeech error: ${JSON.stringify(
                error,
                null,
                2,
              )}`,
            )
          })

        addTestResult('🔄 whisperVadDetectSpeech call initiated (async)')
      } else {
        addTestResult(
          '⚠️ Skipping whisperVadDetectSpeech (VAD context not available)',
        )
      }

      // Test 3: whisperTranscribeData (Promise-based)
      addTestResult('🧪 Testing whisperTranscribeData...')

      const transcribeOptions = {
        language: 'en',
        maxThreads: 2,
        translate: false,
        tokenTimestamps: false,
        tdrzEnable: false,
      }

      global
        .whisperTranscribeData(contextId, transcribeOptions, arrayBuffer)
        .then((result: any) => {
          addTestResult(
            `✅ whisperTranscribeData success: ${JSON.stringify(
              result,
              null,
              2,
            )}`,
          )
        })
        .catch((error: any) => {
          addTestResult(
            `❌ whisperTranscribeData error: ${JSON.stringify(error, null, 2)}`,
          )
        })

      addTestResult('🔄 whisperTranscribeData call initiated (async)')
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
      const whisperContext = await initWhisper({
        filePath: require('../assets/ggml-base.bin'),
      })
      setContextId(whisperContext.id)
      addTestResult(
        `✅ Whisper context initialized with ID: ${whisperContext.id}`,
      )

      // Initialize VAD context (if available)
      try {
        addTestResult('🎤 Initializing VAD context...')
        const vadContext = await initWhisperVad({
          filePath: require('../assets/ggml-silero-v5.1.2.bin'),
        })
        setVadContextId(vadContext.id)
        addTestResult(`✅ VAD context initialized with ID: ${vadContext.id}`)
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
      setContextId(null)
      setVadContextId(null)
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
        {contextId && (
          <Text style={styles.statusSubText}>
            {`Whisper Context ID: ${contextId}`}
          </Text>
        )}
        {vadContextId && (
          <Text style={styles.statusSubText}>
            {`VAD Context ID: ${vadContextId}`}
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
