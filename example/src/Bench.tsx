import React, { useCallback, useRef, useState } from 'react'
import {
  StyleSheet,
  ScrollView,
  View,
  Text,
  Platform,
  Pressable,
} from 'react-native'
import RNFS from 'react-native-fs'
import { initWhisper, libVersion, AudioSessionIos } from '../../src' // whisper.rn
import type { WhisperContext } from '../../src'
import contextOpts from './context-opts'

const baseURL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/'
const modelList = [
  { name: 'tiny', coreml: true },
  { name: 'tiny-q5_1' },
  { name: 'tiny-q8_0' },
  { name: 'base', coreml: true },
  { name: 'base-q5_1' },
  { name: 'base-q8_0' },
  { name: 'small', coreml: true },
  { name: 'small-q5_1' },
  { name: 'small-q8_0' },
  { name: 'medium', coreml: true },
  { name: 'medium-q5_0' },
  { name: 'medium-q8_0' },
  // { name: 'large-v1', coreml: true },
  // { name: 'large-v1-q5_0', },
  // { name: 'large-v1-q8_0', },
  { name: 'large-v2', coreml: true },
  { name: 'large-v2-q5_0' },
  { name: 'large-v2-q8_0' },
  { name: 'large-v3', coreml: true },
  { name: 'large-v3-q5_0' },
  { name: 'large-v3-q8_0' },
  { name: 'large-v3-turbo', coreml: true },
  { name: 'large-v3-turbo-q5_0' },
  { name: 'large-v3-turbo-q8_0' },
] as const

const modelNameMap = modelList.reduce((acc, model) => {
  acc[model.name as keyof typeof acc] = true
  return acc
}, {} as Record<string, boolean>)

const fileDir = `${RNFS.DocumentDirectoryPath}/whisper`

console.log('[App] fileDir', fileDir)

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 10,
  },
  contentContainer: {
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  modelList: {
    padding: 10,
    // auto wrap
    flexWrap: 'wrap',
    flexDirection: 'row',
  },
  modelItem: {
    backgroundColor: '#333',
    borderRadius: 5,
    padding: 5,
    margin: 4,
    flexDirection: 'row',
    alignItems: 'center',
  },
  modelItemUnselected: {
    backgroundColor: '#aaa',
  },
  modelItemText: {
    color: '#ccc',
    fontSize: 12,
    fontWeight: 'bold',
  },
  modelItemTextUnselected: {
    color: '#666',
  },
  button: {
    backgroundColor: '#333',
    borderRadius: 5,
    padding: 8,
    margin: 4,
    width: 'auto',
  },
  buttonText: {
    color: '#ccc',
    fontSize: 12,
    fontWeight: 'bold',
  },
})

export default function Bench() {
  const whisperContextRef = useRef<WhisperContext | null>(null)
  const whisperContext = whisperContextRef.current
  const [logs, setLogs] = useState([])
  const [downloadMap, setDownloadMap] = useState<Record<string, boolean>>(modelNameMap)
  const downloadCount = Object.keys(downloadMap).filter((key) => downloadMap[key]).length
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <Text style={styles.title}>Download List</Text>
      <View style={styles.modelList}>
        {modelList.map((model) => (
          <Pressable
            key={model.name}
            style={[styles.modelItem, !downloadMap[model.name] && styles.modelItemUnselected]}
            onPress={() => {
              setDownloadMap({
                ...downloadMap,
                [model.name]: !downloadMap[model.name],
              })
            }}
          >
            <Text style={styles.modelItemText}>{model.name}</Text>
          </Pressable>
        ))}
      </View>
      <Pressable
        style={styles.button}
        onPress={() => {
        }}
      >
        <Text style={styles.buttonText}>{`Download ${downloadCount} models`}</Text>
      </Pressable>
      <Pressable
        style={styles.button}
        onPress={() => {
        }}
      >
        <Text style={styles.buttonText}>Run benchmark</Text>
      </Pressable>
    </ScrollView>
  )
}
