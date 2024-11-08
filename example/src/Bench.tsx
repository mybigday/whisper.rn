import React, { useCallback, useEffect, useRef, useState } from 'react'
import { StyleSheet, ScrollView, View, Text, Platform } from 'react-native'
import { TouchableOpacity } from 'react-native-gesture-handler'
import RNFS from 'react-native-fs'
import Clipboard from '@react-native-clipboard/clipboard'
import { initWhisper } from '../../src' // whisper.rn
import { createDir, fileDir, modelHost } from './util'
import { Button } from './Button'

const modelList = [
  // TODO: Add coreml model download
  { name: 'tiny', default: true },
  { name: 'tiny-q5_1', default: true },
  { name: 'tiny-q8_0', default: true },
  { name: 'base', default: true },
  { name: 'base-q5_1', default: true },
  { name: 'base-q8_0', default: true },
  { name: 'small', default: true },
  { name: 'small-q5_1', default: true },
  { name: 'small-q8_0', default: true },
  { name: 'medium', default: true },
  { name: 'medium-q5_0', default: true },
  { name: 'medium-q8_0', default: true },
  { name: 'large-v1', default: false },
  { name: 'large-v1-q5_0', default: false },
  { name: 'large-v1-q8_0', default: false },
  { name: 'large-v2', default: false },
  { name: 'large-v2-q5_0', default: false },
  { name: 'large-v2-q8_0', default: false },
  { name: 'large-v3', default: false },
  { name: 'large-v3-q5_0', default: false },
  { name: 'large-v3-q8_0', default: false },
  { name: 'large-v3-turbo', default: false },
  { name: 'large-v3-turbo-q5_0', default: false },
  { name: 'large-v3-turbo-q8_0', default: false },
] as const

const modelNameMap = modelList.reduce((acc, model) => {
  acc[model.name as keyof typeof acc] = model.default
  return acc
}, {} as Record<string, boolean>)

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
  progressBar: {
    backgroundColor: '#3388ff',
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    opacity: 0.5,
    width: '0%',
    borderRadius: 5,
  },
  logContainer: {
    backgroundColor: 'lightgray',
    padding: 8,
    width: '100%',
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
        await createDir(null)
        const { jobId, promise } = RNFS.downloadFile({
          fromUrl: `${modelHost}/ggml-${model.name}.bin?download=true`,
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
    <TouchableOpacity
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
    </TouchableOpacity>
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
    const count = downloadedModelsRef.current.length
    if (
      count > 0 &&
      count === Object.values(downloadMap).filter(Boolean).length
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
      <Text style={styles.title}>Model List</Text>
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
      <Button
        title={
          modelState === 'select'
            ? `Download ${downloadCount} Models`
            : 'Cancel Download'
        }
        onPress={() => {
          if (modelState === 'select') {
            downloadedModelsRef.current = []
            setModelState('download')
          } else {
            setModelState('select')
          }
        }}
      />
      <Button
        title="Start benchmark"
        onPress={async () => {
          log('Start benchmark')
          log(
            '| CPU | OS | Config | Model | Th | FA | Enc. | Dec. | Bch5 | PP | Commit |',
          )
          log(
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
          )
          await Object.entries(downloadMap).reduce(
            async (promise, [modelName, downloadNeeded]) => {
              await promise
              if (!downloadNeeded) return
              const filePath = `${fileDir}/ggml-${modelName}.bin`
              if (!(await RNFS.exists(filePath))) {
                log(`${modelName} not found, skipping`)
                return
              }
              const ctx = await initWhisper({
                filePath,
                useCoreMLIos: false,
                useGpu: Platform.OS === 'ios',
                useFlashAttn: Platform.OS === 'ios',
              })
              try {
                const result = await ctx.bench(-1)
                const { config, nThreads, nEncode, nDecode, nBatchd, nPrompt } =
                  result
                const fa = Platform.OS === 'ios' ? '1' : '0'
                const systemInfo = config
                  .split(' ')
                  .filter((c) => ['NEON', 'BLAS', 'METAL'].includes(c))
                  .join(' ')
                log(
                  `| <todo> | ${
                    Platform.OS
                  } | ${systemInfo} | ${modelName} | ${nThreads} | ${fa} | ${nEncode.toFixed(
                    2,
                  )} | ${nDecode.toFixed(2)} | ${nBatchd.toFixed(
                    2,
                  )} | ${nPrompt.toFixed(2)} | <todo> |`,
                )
              } finally {
                await ctx.release()
              }
            },
            Promise.resolve(),
          )
        }}
      />
      <View style={styles.logContainer}>
        {logs.map((msg, index) => (
          <Text key={index} style={styles.logText}>
            {msg}
          </Text>
        ))}
      </View>
      <View style={styles.buttonContainer}>
        <Button
          title="Copy Logs"
          onPress={() => Clipboard.setString(logs.join('\n'))}
        />
        <Button title="Clear Logs" onPress={() => setLogs([])} />
        <Button
          title="Delete Models"
          onPress={() => {
            setModelState('select')
            RNFS.readDir(fileDir).then((files) => {
              files.forEach((file) => {
                if (file.name.startsWith('ggml-')) RNFS.unlink(file.path)
              })
            })
          }}
        />
      </View>
    </ScrollView>
  )
}
