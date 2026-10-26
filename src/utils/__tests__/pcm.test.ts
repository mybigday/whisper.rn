import { floatToPcm16, pcmPeak, toPcm16ArrayBuffer } from '../pcm'

test('converts normalized float samples to signed 16-bit PCM', () => {
  const result = floatToPcm16(
    new Float32Array([0, 0.5, -0.5, 1, -1, 2, -2]),
  )

  expect(Array.from(result)).toEqual([
    0, 16384, -16384, 32767, -32768, 32767, -32768,
  ])
})

test('copies only the selected Int16Array range', () => {
  const input = new Int16Array([1, 2, 3, 4])
  const result = toPcm16ArrayBuffer(input.subarray(1, 3))

  expect(Array.from(new Int16Array(result))).toEqual([2, 3])
})

test('rejects an odd-length PCM buffer', () => {
  expect(() => toPcm16ArrayBuffer(new ArrayBuffer(3))).toThrow(
    'expected signed 16-bit PCM but the buffer has an odd byte length',
  )
})

test('reports the peak amplitude of signed 16-bit PCM', () => {
  expect(pcmPeak(new Int16Array([0, -32768, 16384]).buffer)).toBe(1)
  expect(pcmPeak(new Int16Array([0, 0, 0]).buffer)).toBe(0)
})
