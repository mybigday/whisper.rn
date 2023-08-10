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

  const { subscribe } = await context.transcribeRealtime()
  const events: any[] = []
  subscribe((event) => events.push(event))
  await new Promise((resolve) => setTimeout(resolve, 0))
  expect(events).toMatchSnapshot()
  await context.release()
  await releaseAllWhisper()
})
