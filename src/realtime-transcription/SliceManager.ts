import type { AudioSlice, MemoryUsage } from './types'

/** Initial slice buffer: 1 s of 16 kHz PCM16. It doubles as needed. */
const INITIAL_SLICE_BYTES = 16000 * 2

/**
 * Append into the slice's buffer, doubling it when full (amortized O(1) per byte).
 * Views handed out earlier cover only bytes already written, so appending never
 * changes them: a grown buffer is a new one, and the old views keep the old one.
 */
function appendToSlice(slice: AudioSlice, data: Uint8Array): void {
  const needed = slice.sampleCount + data.length
  if (slice.data.length < needed) {
    // The view may have been trimmed (finalized): widen it over its buffer if that has room
    const { buffer, byteOffset } = slice.data
    const room = buffer.byteLength - byteOffset
    if (room >= needed) {
      slice.data = new Uint8Array(buffer, byteOffset, room)
    } else {
      const grown = new Uint8Array(Math.max(needed, 2 * room, INITIAL_SLICE_BYTES))
      grown.set(slice.data.subarray(0, slice.sampleCount))
      slice.data = grown
    }
  }
  slice.data.set(data, slice.sampleCount)
  slice.sampleCount += data.length
  slice.endTime = Date.now()
}

/**
 * Audio slices of a realtime session: one per speech segment, closed by
 * `forceNextSlice()`. Slices grow as audio arrives (their length is bounded by
 * the segmenter's maximum duration); only the newest `maxSlicesInMemory` keep
 * their audio.
 *
 * Note: `sampleCount` counts bytes of PCM16, not samples.
 */
export class SliceManager {
  private slices: AudioSlice[] = []

  private currentSliceIndex = 0

  private transcribeSliceIndex = 0

  private maxSlicesInMemory: number

  constructor(maxSlicesInMemory = 3) {
    this.maxSlicesInMemory = Math.max(1, maxSlicesInMemory)
  }

  /**
   * Add audio data to the current slice
   */
  addAudioData(audioData: Uint8Array): { slice: AudioSlice } {
    const slice = this.getCurrentSlice()
    if (audioData.length > 0) appendToSlice(slice, audioData)
    return { slice }
  }

  private getCurrentSlice(): AudioSlice {
    let slice = this.slices.find((s) => s.index === this.currentSliceIndex)

    if (!slice) {
      slice = {
        index: this.currentSliceIndex,
        data: new Uint8Array(0),
        sampleCount: 0,
        startTime: Date.now(),
        endTime: Date.now(),
        isProcessed: false,
        isReleased: false,
      }
      this.slices.push(slice)

      // Clean up old slices if we have too many
      this.cleanupOldSlices()
    }

    return slice
  }

  /**
   * Finalize the current slice
   */
  private finalizeCurrentSlice(): void {
    const slice = this.slices.find((s) => s.index === this.currentSliceIndex)
    if (slice && slice.sampleCount > 0) {
      // Trim the data array to actual size
      slice.data = slice.data.subarray(0, slice.sampleCount)
      slice.endTime = Date.now()
    }
  }

  /**
   * Get a slice for transcription
   */
  getSliceForTranscription(): AudioSlice | null {
    const slice = this.slices.find(
      (s) => s.index === this.transcribeSliceIndex && !s.isProcessed,
    )

    if (slice && slice.sampleCount > 0) {
      return slice
    }

    return null
  }

  /**
   * Mark a slice as processed
   */
  markSliceAsProcessed(sliceIndex: number): void {
    const slice = this.slices.find((s) => s.index === sliceIndex)
    if (slice) {
      slice.isProcessed = true
    }
  }

  /**
   * Move to the next slice for transcription
   */
  moveToNextTranscribeSlice(): void {
    this.transcribeSliceIndex += 1
  }

  /**
   * The audio of a slice (a view of exactly its bytes)
   */
  getAudioDataForTranscription(sliceIndex: number): Uint8Array | null {
    const slice = this.slices.find((s) => s.index === sliceIndex)
    if (!slice || slice.sampleCount === 0) {
      return null
    }

    return slice.data.subarray(0, slice.sampleCount)
  }

  /**
   * Get a slice by index
   */
  getSliceByIndex(sliceIndex: number): AudioSlice | null {
    return this.slices.find((s) => s.index === sliceIndex) || null
  }

  /**
   * Clean up old slices to manage memory
   */
  private cleanupOldSlices(): void {
    if (this.slices.length <= this.maxSlicesInMemory) {
      return
    }

    // Sort slices by index
    this.slices.sort((a, b) => a.index - b.index)

    // Keep only the most recent slices
    const slicesToKeep = this.slices.slice(-this.maxSlicesInMemory)
    const slicesToRemove = this.slices.slice(0, -this.maxSlicesInMemory)

    // Release old slices
    slicesToRemove.forEach((slice) => {
      if (!slice.isReleased) {
        slice.isReleased = true
        // Clear the audio data to free memory
        slice.data = new Uint8Array(0)
      }
    })

    this.slices = slicesToKeep
  }

  /**
   * Get memory usage statistics
   */
  getMemoryUsage(): MemoryUsage {
    const activeSlices = this.slices.filter((s) => !s.isReleased)
    const totalBytes = activeSlices.reduce(
      (sum, slice) => sum + slice.sampleCount,
      0,
    )

    // Estimate memory usage (Uint8Array = 1 byte per sample)
    const estimatedMB = totalBytes / (1024 * 1024)

    return {
      slicesInMemory: activeSlices.length,
      totalSamples: totalBytes / 2, // Convert bytes to samples (2 bytes per sample)
      estimatedMB: Math.round(estimatedMB * 100) / 100, // Round to 2 decimal places
    }
  }

  /**
   * Reset all slices and indices
   */
  reset(): void {
    // Release all slices
    this.slices.forEach((slice) => {
      slice.isReleased = true
      // Clear the audio data to free memory
      slice.data = new Uint8Array(0)
    })

    // Reset state
    this.slices = []
    this.currentSliceIndex = 0
    this.transcribeSliceIndex = 0
  }

  /**
   * Get current slice information
   */
  getCurrentSliceInfo() {
    return {
      currentSliceIndex: this.currentSliceIndex,
      transcribeSliceIndex: this.transcribeSliceIndex,
      totalSlices: this.slices.length,
      memoryUsage: this.getMemoryUsage(),
    }
  }

  /**
   * Shorten the current slice to its first `byteLength` bytes (no-op if it is
   * not longer), e.g. when its speech segment turns out to end earlier.
   */
  truncateCurrentSlice(byteLength: number): void {
    const slice = this.slices.find((s) => s.index === this.currentSliceIndex)
    if (slice && byteLength < slice.sampleCount) {
      slice.sampleCount = Math.max(0, byteLength - (byteLength % 2))
    }
  }

  /**
   * Close the current slice and move to the next one
   */
  forceNextSlice(): { slice?: AudioSlice } {
    const currentSlice = this.slices.find(
      (s) => s.index === this.currentSliceIndex,
    )

    if (currentSlice && currentSlice.sampleCount > 0) {
      // Finalize current slice
      this.finalizeCurrentSlice()

      // Move to next slice
      this.currentSliceIndex += 1

      return { slice: currentSlice }
    }

    // If no current slice or it's empty, just move to next index
    this.currentSliceIndex += 1
    return {}
  }
}
