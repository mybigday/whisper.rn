const clamp = (value: number): number => Math.max(-1, Math.min(1, value))

export function floatToPcm16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i += 1) {
    const sample = clamp(samples[i] ?? 0)
    out[i] = Math.round(sample < 0 ? sample * 0x8000 : sample * 0x7fff)
  }
  return out
}

export function toPcm16ArrayBuffer(
  data: ArrayBuffer | Int16Array | Float32Array,
): ArrayBuffer {
  if (data instanceof Float32Array) {
    return floatToPcm16(data).buffer as ArrayBuffer
  }

  const buffer: ArrayBuffer =
    data instanceof Int16Array
      ? (data.buffer.slice(
          data.byteOffset,
          data.byteOffset + data.byteLength,
        ) as ArrayBuffer)
      : data

  if (buffer.byteLength % 2 !== 0) {
    throw new Error(
      'expected signed 16-bit PCM but the buffer has an odd byte length',
    )
  }

  return buffer
}

export function pcmPeak(pcm: ArrayBuffer): number {
  const view = new Int16Array(pcm)
  let peak = 0
  for (let i = 0; i < view.length; i += 1) {
    const magnitude = Math.abs(view[i] ?? 0)
    if (magnitude > peak) peak = magnitude
  }
  return peak / 0x8000
}
