import { initWhisper, releaseAllWhisper } from '..'

jest.mock('..', () => require('../../jest/mock'))

test('Mock', async () => {
  const context = await initWhisper()
  expect(context.id).toBe(1)
  expect(await context.transcribe('test.wav')).toEqual({
    result: ' Test',
    segments: [{ text: ' Test', t0: 0, t1: 33 }],
  })
  await context.release()
  await releaseAllWhisper()
})
