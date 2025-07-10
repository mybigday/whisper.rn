import { initWhisper, releaseAllWhisper } from '..'

jest.mock('..', () => require('../../jest/mock'))

Math.random = () => 0.5

test('Mock', async () => {
  const context = await initWhisper({
    filePath: 'test.bin',
  })
  expect(context.id).toBe(1)
  const { promise } = context.transcribe('test.wav')
  expect(await promise).toEqual({
    isAborted: false,
    result: ' Test',
    segments: [{ text: ' Test', t0: 0, t1: 33 }],
  })
  await context.release()
  await releaseAllWhisper()
})
