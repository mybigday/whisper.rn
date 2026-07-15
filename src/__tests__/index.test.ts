import {
  initParakeet,
  initWhisper,
  releaseAllParakeet,
  releaseAllWhisper,
} from '..'
import type { ParakeetContext } from '..'
import type { ParakeetContextLike } from '../realtime-transcription/types'

jest.mock('..', () => require('../jest-mock'))

Math.random = () => 0.5

type ParakeetContextCompatibility =
  ParakeetContext extends ParakeetContextLike ? true : false

const parakeetContextIsRealtimeCompatible: ParakeetContextCompatibility = true

const parakeetMocks = {
  init: global.parakeetInitContext as jest.MockedFunction<
    typeof global.parakeetInitContext
  >,
  release: global.parakeetReleaseContext as jest.MockedFunction<
    typeof global.parakeetReleaseContext
  >,
  releaseAll: global.parakeetReleaseAllContexts as jest.MockedFunction<
    typeof global.parakeetReleaseAllContexts
  >,
  transcribeFile: global.parakeetTranscribeFile as jest.MockedFunction<
    typeof global.parakeetTranscribeFile
  >,
  transcribeData: global.parakeetTranscribeData as jest.MockedFunction<
    typeof global.parakeetTranscribeData
  >,
  abort: global.parakeetAbortTranscribe as jest.MockedFunction<
    typeof global.parakeetAbortTranscribe
  >,
}

beforeEach(() => {
  jest.clearAllMocks()
})

test('provides the Whisper mock API', async () => {
  const context = await initWhisper({
    filePath: 'test.bin',
  })
  expect(context.id).toBe(1)
  const { promise } = context.transcribe('test.wav')
  expect(await promise).toEqual({
    language: 'en',
    isAborted: false,
    result: ' Test',
    segments: [{ text: ' Test', t0: 0, t1: 33 }],
  })
  await context.release()
  await releaseAllWhisper()
})

test('initializes and releases a Parakeet context', async () => {
  expect(parakeetContextIsRealtimeCompatible).toBe(true)

  const context = await initParakeet({
    filePath: 'file:///models/parakeet.bin',
    isBundleAsset: true,
    useGpu: false,
  })

  expect(parakeetMocks.init).toHaveBeenCalledWith(context.id, {
    filePath: '/models/parakeet.bin',
    isBundleAsset: true,
    useGpu: false,
  })
  expect(context).toMatchObject({
    gpu: false,
    reasonNoGPU: 'Mock Parakeet context',
  })

  await context.release()
  expect(parakeetMocks.release).toHaveBeenCalledWith(context.id)

  await releaseAllParakeet()
  expect(parakeetMocks.releaseAll).toHaveBeenCalledTimes(1)
})

test('transcribes Parakeet audio files and aborts the matching job', async () => {
  const context = await initParakeet({ filePath: 'parakeet.bin' })
  const task = context.transcribe('file:///audio/jfk.wav', {
    maxThreads: 3,
    audioCtx: 1500,
  })

  expect(parakeetMocks.transcribeFile).toHaveBeenCalledTimes(1)
  const [contextId, path, options] = parakeetMocks.transcribeFile.mock.calls[0]!
  expect(contextId).toBe(context.id)
  expect(path).toBe('/audio/jfk.wav')
  expect(options).toMatchObject({ maxThreads: 3, audioCtx: 1500 })
  expect(options.jobId).toEqual(expect.any(Number))
  await expect(task.promise).resolves.toEqual({
    language: '',
    result: ' Parakeet test',
    segments: [{ text: ' Parakeet test', t0: 0, t1: 1101 }],
    isAborted: false,
  })

  await task.stop()
  expect(parakeetMocks.abort).toHaveBeenCalledWith(context.id, options.jobId)
})

test('transcribes Parakeet ArrayBuffer and base64 audio data', async () => {
  const context = await initParakeet({ filePath: 'parakeet.bin' })
  const audioData = new Int16Array([0, 8192, -8192]).buffer

  await context.transcribeData(audioData, { maxThreads: 2 }).promise
  expect(parakeetMocks.transcribeData).toHaveBeenLastCalledWith(
    context.id,
    expect.objectContaining({ maxThreads: 2, jobId: expect.any(Number) }),
    audioData,
  )

  await context.transcribeData('AAAAAA==').promise
  const decodedData = parakeetMocks.transcribeData.mock.calls[1]![2]
  expect(Array.from(new Uint8Array(decodedData))).toEqual([0, 0, 0, 0])
})

test('rejects remote Parakeet models and audio files', async () => {
  await expect(
    initParakeet({ filePath: 'https://example.com/parakeet.bin' }),
  ).rejects.toThrow('Remote Parakeet models are not supported')

  const context = await initParakeet({ filePath: 'parakeet.bin' })
  expect(() => context.transcribe('https://example.com/jfk.wav')).toThrow(
    'Parakeet remote audio files are not supported',
  )
})
