import { SliceManager } from '../SliceManager'

// Regression tests for memory growth in realtime transcription on slow devices.

describe('SliceManager', () => {
  it('stores data of any size in the current slice', () => {
    const manager = new SliceManager(3)
    const size = 32000 * 3 + 100
    manager.addAudioData(new Uint8Array(size))
    expect(manager.getAudioDataForTranscription(0)!.length).toBe(size)
  })

  it('grows the buffer geometrically, keeping earlier views intact', () => {
    const manager = new SliceManager(3)
    const buffers = new Set<ArrayBufferLike>()
    const views: Uint8Array[] = []
    // 60 s in 100 ms chunks
    for (let i = 0; i < 600; i += 1) {
      manager.addAudioData(new Uint8Array(3200).fill(i % 251))
      const view = manager.getAudioDataForTranscription(0)!
      buffers.add(view.buffer)
      views.push(view)
    }
    // 1 s doubling to 60 s: a handful of buffers, not one per chunk
    expect(buffers.size).toBeLessThanOrEqual(8)
    expect(views[9]!.length).toBe(32000)
    expect(views[9]![31999]).toBe(9)
    expect(manager.getAudioDataForTranscription(0)!.length).toBe(600 * 3200)
  })

  it('truncates the current slice and keeps only the newest slices', () => {
    const manager = new SliceManager(2)
    for (let i = 0; i < 4; i += 1) {
      manager.addAudioData(new Uint8Array(3200).fill(i + 1))
      if (i === 3) manager.truncateCurrentSlice(1001)
      if (i < 3) manager.forceNextSlice()
    }
    expect(manager.getSliceByIndex(0)).toBeNull()
    expect(manager.getAudioDataForTranscription(3)!.length).toBe(1000)
    expect(manager.getMemoryUsage().slicesInMemory).toBeLessThanOrEqual(2)
  })
})
