import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  StyleSheet,
  ScrollView,
  View,
  Text,
  Platform,
  Pressable,
} from 'react-native'
import RNFS from 'react-native-fs'
import Clipboard from '@react-native-clipboard/clipboard'
import { initWhisper } from '../../src' // whisper.rn
import contextOpts from './context-opts'

const baseURL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/'
const modelList = [
  // TODO: Add coreml model download
  { name: 'tiny' },
  { name: 'tiny-q5_1' },
  { name: 'tiny-q8_0' },
  { name: 'base' },
  { name: 'base-q5_1' },
  { name: 'base-q8_0' },
  { name: 'small' },
  { name: 'small-q5_1' },
  { name: 'small-q8_0' },
  { name: 'medium' },
  { name: 'medium-q5_0' },
  { name: 'medium-q8_0' },
  { name: 'large-v1' },
  { name: 'large-v1-q5_0' },
  { name: 'large-v1-q8_0' },
  { name: 'large-v2' },
  { name: 'large-v2-q5_0' },
  { name: 'large-v2-q8_0' },
  { name: 'large-v3' },
  { name: 'large-v3-q5_0' },
  { name: 'large-v3-q8_0' },
  { name: 'large-v3-turbo' },
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
    margin: 4,
    flexDirection: 'row',
    alignItems: 'center',
  },
  modelItemUnselected: {
    backgroundColor: '#aaa',
  },
  modelItemText: {
    margin: 6,
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
  progressBar: {
    backgroundColor: '#3388ff',
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    opacity: 0.5,
    width: '0%',
  },
  logContainer: {
    backgroundColor: 'lightgray',
    padding: 8,
    width: '95%',
    borderRadius: 8,
    marginVertical: 8,
  },
  logText: { fontSize: 12, color: '#333' },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
  },
})

const Model = (props: {
  model: (typeof modelList)[number]
  state: 'select' | 'download'
  downloadMap: Record<string, boolean>
  setDownloadMap: (downloadMap: Record<string, boolean>) => void
  onDownloadStarted: (modelName: string) => void
  onDownloaded: (modelName: string) => void
}) => {
  const {
    model,
    state,
    downloadMap,
    setDownloadMap,
    onDownloadStarted,
    onDownloaded,
  } = props

  const downloadRef = useRef<number | null>(null)
  const [progress, setProgress] = useState(0)

  const downloadNeeded = downloadMap[model.name]

  const cancelDownload = async () => {
    if (downloadRef.current) {
      RNFS.stopDownload(downloadRef.current)
      downloadRef.current = null
      setProgress(0)
    }
  }

  useEffect(() => {
    if (state !== 'select') return
    RNFS.exists(`${fileDir}/ggml-${model.name}.bin`).then((exists) => {
      if (exists) setProgress(1)
      else setProgress(0)
    })
  }, [model.name, state])

  useEffect(() => {
    if (state === 'download') {
      const download = async () => {
        if (!downloadNeeded) return cancelDownload()
        if (await RNFS.exists(`${fileDir}/ggml-${model.name}.bin`)) {
          setProgress(1)
          onDownloaded(model.name)
          return
        }
        const { jobId, promise } = RNFS.downloadFile({
          fromUrl: `${baseURL}ggml-${model.name}.bin?download=true`,
          toFile: `${fileDir}/ggml-${model.name}.bin`,
          begin: () => {
            setProgress(0)
            onDownloadStarted(model.name)
          },
          progress: (res) => {
            setProgress(res.bytesWritten / res.contentLength)
          },
        })
        downloadRef.current = jobId
        promise.then(() => {
          setProgress(1)
          onDownloaded(model.name)
        })
      }
      download()
    } else {
      cancelDownload()
    }
  }, [state, downloadNeeded, model.name, onDownloadStarted, onDownloaded])

  return (
    <Pressable
      key={model.name}
      style={[
        styles.modelItem,
        !downloadMap[model.name] && styles.modelItemUnselected,
      ]}
      onPress={() => {
        if (downloadRef.current) {
          cancelDownload()
          return
        }
        if (state === 'download') return
        setDownloadMap({
          ...downloadMap,
          [model.name]: !downloadMap[model.name],
        })
      }}
    >
      <Text style={styles.modelItemText}>{model.name}</Text>

      {downloadNeeded && (
        <View style={[styles.progressBar, { width: `${progress * 100}%` }]} />
      )}
    </Pressable>
  )
}

export default function Bench() {
  const [logs, setLogs] = useState<string[]>([])
  const [downloadMap, setDownloadMap] =
    useState<Record<string, boolean>>(modelNameMap)
  const [modelState, setModelState] = useState<'select' | 'download'>('select')

  const downloadedModelsRef = useRef<string[]>([])

  const log = useCallback((...messages: any[]) => {
    setLogs((prev) => [...prev, messages.join(' ')])
  }, [])

  useEffect(() => {
    if (
      downloadedModelsRef.current.length ===
      Object.values(downloadMap).filter(Boolean).length
    ) {
      downloadedModelsRef.current = []
      setModelState('select')
      log('All models downloaded')
    }
  }, [log, logs, downloadMap])

  const handleDownloadStarted = useCallback(
    (modelName: string) => {
      log(`Downloading ${modelName}`)
    },
    [log],
  )

  const handleDownloaded = useCallback(
    (modelName: string) => {
      downloadedModelsRef.current = [...downloadedModelsRef.current, modelName]
      log(`Downloaded ${modelName}`)
    },
    [log],
  )

  const downloadCount = Object.keys(downloadMap).filter(
    (key) => downloadMap[key],
  ).length
  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
    >
      <Text style={styles.title}>Download List</Text>
      <View style={styles.modelList}>
        {modelList.map((model) => (
          <Model
            key={model.name}
            state={modelState}
            model={model}
            downloadMap={downloadMap}
            setDownloadMap={setDownloadMap}
            onDownloadStarted={handleDownloadStarted}
            onDownloaded={handleDownloaded}
          />
        ))}
      </View>
      <Pressable
        style={styles.button}
        onPress={() => {
          if (modelState === 'select') {
            downloadedModelsRef.current = []
            setModelState('download')
          } else {
            setModelState('select')
          }
        }}
      >
        <Text style={styles.buttonText}>
          {`${
            modelState === 'select' ? 'Download' : 'Cancel'
          } ${downloadCount} models`}
        </Text>
      </Pressable>
      <Pressable
        style={styles.button}
        onPress={async () => {
          log('Start benchmark')
          log(
            '| CPU | OS | Config | Model | Th | FA | Enc. | Dec. | Bch5 | PP | Commit |',
          )
          log(
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
          )
          await Object.entries(downloadMap).reduce(
            async (acc, [modelName, downloadNeeded]) => {
              if (!downloadNeeded) return acc
              const filePath = `${fileDir}/ggml-${modelName}.bin`
              if (!(await RNFS.exists(filePath))) {
                log(`${modelName} not found, skipping`)
                return acc
              }
              const ctx = await initWhisper({
                filePath,
                useCoreMLIos: false,
                useGpu: Platform.OS === 'ios',
                useFlashAttn: Platform.OS === 'ios',
              })
              try {
                const result = await ctx.bench(-1)
                const { nThreads, nEncode, nDecode, nBatchd, nPrompt } = result
                const fa = Platform.OS === 'ios' ? '1' : '0'
                log(
                  // TODO: config
                  `| <todo> | ${
                    Platform.OS
                  } | NEON BLAS METAL | ${modelName} | ${nThreads} | ${fa} | ${nEncode.toFixed(
                    2,
                  )} | ${nDecode.toFixed(2)} | ${nBatchd.toFixed(
                    2,
                  )} | ${nPrompt.toFixed(2)} | <todo> |`,
                )
              } finally {
                await ctx.release()
              }
              return acc
            },
            Promise.resolve(),
          )
        }}
      >
        <Text style={styles.buttonText}>Run benchmark</Text>
      </Pressable>
      <Pressable
        style={({ pressed }: { pressed: boolean }) => [
          styles.logContainer,
          { backgroundColor: pressed ? '#ccc' : 'lightgray', opacity: pressed ? 0.75 : 1 },
        ]}
        onPress={() => Clipboard.setString(logs.join('\n'))}
      >
        {logs.map((msg, index) => (
          <Text key={index} style={styles.logText}>
            {msg}
          </Text>
        ))}
      </Pressable>
      <View style={styles.buttonContainer}>
        <Pressable style={styles.button} onPress={() => setLogs([])}>
          <Text style={styles.buttonText}>Clear Logs</Text>
        </Pressable>
        <Pressable
          style={styles.button}
          onPress={() => {
            setModelState('select')
            RNFS.readDir(fileDir).then((files) => {
              files.forEach((file) => {
                if (file.name.startsWith('ggml-')) RNFS.unlink(file.path)
              })
            })
          }}
        >
          <Text style={styles.buttonText}>Clear Downloaded Models</Text>
        </Pressable>
      </View>
    </ScrollView>
  )
}
