import { initWhisper, releaseAllWhisper } from '..'

jest.mock('..', () => require('../../jest/mock'))

test('Mock', async () => {
  const context = await initWhisper()
  expect(context.id).toBe(1)
  expect(await context.transcribe('test.wav')).toEqual({ result: 'TEST' })
  await context.release()
  await releaseAllWhisper()
})
