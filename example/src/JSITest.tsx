import * as React from 'react'
import { View, Text, StyleSheet, ScrollView } from 'react-native'
import { TouchableOpacity } from 'react-native-gesture-handler'
import { initWhisper, releaseAllWhisper, installJSIBindings } from '../../src'

// Declare global JSI functions
declare global {
  // eslint-disable-next-line no-var
  var whisperTestContext: ((contextId: number) => boolean) | undefined
}

const styles = StyleSheet.create({
  jsiContainer: {
    flex: 1,
    padding: 20,
  },
  jsiButton: {
    margin: 5,
    padding: 15,
    backgroundColor: '#007AFF',
    borderRadius: 8,
    alignItems: 'center',
  },
  jsiButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  logContainer: {
    flex: 1,
    marginTop: 20,
    padding: 10,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
  },
  logTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  logText: {
    fontSize: 12,
    fontFamily: 'monospace',
    color: '#333',
  },
})

function JSITestScreen() {
  const [logs, setLogs] = React.useState<string[]>([])
  const [contextId, setContextId] = React.useState<number | null>(null)
  const [jsiInstalled, setJsiInstalled] = React.useState(false)

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`])
  }

  const initializeContext = async () => {
    try {
      addLog('Initializing whisper context...')
      const result = await initWhisper({
        filePath: require('../assets/ggml-base.bin'),
      })
      setContextId(result.id)
      addLog(`✅ Context initialized with ID: ${result.id}`)
    } catch (error) {
      addLog(`❌ Failed to initialize context: ${error}`)
    }
  }

  const releaseContext = async () => {
    try {
      addLog('Releasing all contexts...')
      await releaseAllWhisper()
      setContextId(null)
      addLog('✅ All contexts released')
    } catch (error) {
      addLog(`❌ Failed to release contexts: ${error}`)
    }
  }

    const installJSI = async () => {
    try {
      addLog('Installing JSI bindings...')
      await installJSIBindings()
      setJsiInstalled(true)
      addLog(`✅ JSI bindings installed`)
    } catch (error) {
      addLog(`❌ Failed to install JSI bindings: ${error}`)
    }
  }

  const testJSIFunction = () => {
    try {
      addLog('Testing JSI function...')

      // Check if global JSI functions are available
      if (typeof global.whisperTestContext === 'function') {
        const result = global.whisperTestContext(contextId || 1)
        addLog(`✅ JSI function result: ${result}`)
      } else {
        addLog('❌ JSI function not available (whisperTestContext)')
      }
    } catch (error) {
      addLog(`❌ JSI function error: ${error}`)
    }
  }

  const clearLogs = () => {
    setLogs([])
  }

  const contextText = contextId ? `(ID: ${contextId})` : ''
  const jsiStatusText = jsiInstalled ? '✅' : ''

  return (
    <View style={styles.jsiContainer}>
      <TouchableOpacity style={styles.jsiButton} onPress={initializeContext}>
        <Text style={styles.jsiButtonText}>
          {`Initialize Whisper Context ${contextText}`}
        </Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.jsiButton} onPress={releaseContext}>
        <Text style={styles.jsiButtonText}>Release All Contexts</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.jsiButton} onPress={installJSI}>
        <Text style={styles.jsiButtonText}>
          {`Install JSI Bindings ${jsiStatusText}`}
        </Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.jsiButton} onPress={testJSIFunction}>
        <Text style={styles.jsiButtonText}>Test JSI Function</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.jsiButton, { backgroundColor: '#FF3B30' }]}
        onPress={clearLogs}
      >
        <Text style={styles.jsiButtonText}>Clear Logs</Text>
      </TouchableOpacity>

      <View style={styles.logContainer}>
        <Text style={styles.logTitle}>Logs:</Text>
        <ScrollView>
          {logs.map((log, index) => (
            <Text key={index} style={styles.logText}>
              {log}
            </Text>
          ))}
        </ScrollView>
      </View>
    </View>
  )
}

export default JSITestScreen
