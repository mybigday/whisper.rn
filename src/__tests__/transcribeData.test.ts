import { initWhisper } from '..'

jest.mock('..', () => require('../jest-mock'))

const transcribeData = global.whisperTranscribeData as jest.MockedFunction<
  typeof global.whisperTranscribeData
>

beforeEach(() => {
  jest.clearAllMocks()
})

test('converts Float32Array input before calling native code', async () => {
  const context = await initWhisper({ filePath: 'test.bin' })

  await context.transcribeData(new Float32Array([0, 0.5, -1])).promise

  expect(transcribeData).toHaveBeenCalledTimes(1)
  const audioData = transcribeData.mock.calls[0]![2]
  expect(Array.from(new Int16Array(audioData))).toEqual([0, 16384, -32768])
})

test('rejects an odd-length ArrayBuffer', async () => {
  const context = await initWhisper({ filePath: 'test.bin' })

  expect(() => context.transcribeData(new ArrayBuffer(3))).toThrow(
    'expected signed 16-bit PCM but the buffer has an odd byte length',
  )
})

test('rejects an invalid peak threshold', async () => {
  const context = await initWhisper({ filePath: 'test.bin' })

  expect(() =>
    context.transcribeData(new Int16Array([1]), { peakThreshold: 2 }),
  ).toThrow('peakThreshold must be between 0 and 1')
})

test('drops a silent buffer without calling native code', async () => {
  const context = await initWhisper({ filePath: 'test.bin' })

  const task = context.transcribeData(new Int16Array([0, 0, 0]), {
    peakThreshold: 0.01,
  })
  const result = await task.promise

  expect(transcribeData).not.toHaveBeenCalled()
  expect(result).toEqual({
    result: '',
    language: '',
    segments: [],
    isAborted: false,
    skipped: 'silence',
  })
  await task.stop()
})

test('runs inference when the buffer is above the peak threshold', async () => {
  const context = await initWhisper({ filePath: 'test.bin' })

  await context.transcribeData(new Int16Array([16384]), {
    peakThreshold: 0.01,
  }).promise

  expect(transcribeData).toHaveBeenCalledTimes(1)
  expect(transcribeData.mock.calls[0]![1]).not.toHaveProperty(
    'peakThreshold',
  )
})
