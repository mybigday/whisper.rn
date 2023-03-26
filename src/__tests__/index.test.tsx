import { initWhisper, releaseAllWhisper } from '..'

jest.mock('..', () => require('../../jest/mock'))

test('Mock', async () => {
  const context = await initWhisper()
  expect(context.id).toBe(1)
  const { promise } = context.transcribe('test.wav')
  expect(await promise).toEqual({
    result: ' Test',
    segments: [{ text: ' Test', t0: 0, t1: 33 }],
  })

  const { subscribe } = await context.transcribeRealtime()
  const events: any[] = []
  subscribe((event) => events.push(event))
  await new Promise((resolve) => setTimeout(resolve, 0))
  expect(events).toMatchObject([
    {
      contextId: 1,
      data: {
        result: ' Test',
        segments: [
          {
            t0: 0,
            t1: 33,
            text: ' Test',
          },
        ],
      },
      isCapturing: true,
      processTime: 100,
      recordingTime: 1000,
    },
    {
      contextId: 1,
      data: {
        result: ' Test',
        segments: [
          {
            t0: 0,
            t1: 33,
            text: ' Test',
          },
        ],
      },
      isCapturing: false,
      processTime: 100,
      recordingTime: 2000,
    },
  ])
  await context.release()
  await releaseAllWhisper()
})
