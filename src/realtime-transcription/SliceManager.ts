import type { AudioSlice, MemoryUsage } from './types'

/**
 * Append into the slice's preallocated buffer. The old code copied the whole slice into a new
 * array on every chunk, so each queued transcription held its own full copy.
 * Views handed out earlier cover only bytes already written, so appending never changes them.
 */
function appendToSlice(slice: AudioSlice, data: Uint8Array, bytesPerSlice: number): void {
  if (slice.data.length < slice.sampleCount + data.length) {
    // The buffer was trimmed by finalizeCurrentSlice(); give it room again
    const grown = new Uint8Array(Math.max(bytesPerSlice, slice.sampleCount + data.length))
    grown.set(slice.data.subarray(0, slice.sampleCount))
    slice.data = grown
  }
  slice.data.set(data, slice.sampleCount)
  slice.sampleCount += data.length
  slice.endTime = Date.now()
}

export class SliceManager {
  private slices: AudioSlice[] = []

  private currentSliceIndex = 0

  private transcribeSliceIndex = 0

  private maxSlicesInMemory: number

  private sliceDurationSec: number

  private sampleRate: number

  constructor(
    sliceDurationSec = 30,
    maxSlicesInMemory = 1,
    sampleRate = 16000,
  ) {
    this.sliceDurationSec = sliceDurationSec
    this.maxSlicesInMemory = maxSlicesInMemory
    this.sampleRate = sampleRate
  }

  /**
   * Add audio data to the current slice
   */
  addAudioData(audioData: Uint8Array): {
    slice?: AudioSlice
  } {
    // Calculate bytes per slice (2 bytes per sample for 16-bit PCM)
    const bytesPerSlice = this.sliceDurationSec * this.sampleRate * 2

    let remaining = audioData
    let currentSlice = this.getCurrentSlice()

    // Iterative rather than recursive: data larger than a whole slice used to recurse without end,
    // creating a new slice buffer on every call.
    while (remaining.length > 0) {
      if (currentSlice.sampleCount + remaining.length <= bytesPerSlice) {
        appendToSlice(currentSlice, remaining, bytesPerSlice)
        remaining = remaining.subarray(remaining.length)
      } else if (currentSlice.sampleCount > 0) {
        // Finalize current slice and continue in a new one (the chunk is not split, as before)
        this.finalizeCurrentSlice()
        this.currentSliceIndex += 1
        currentSlice = this.getCurrentSlice()
      } else {
        // Even an empty slice cannot hold it: fill this slice and carry the rest over
        appendToSlice(currentSlice, remaining.subarray(0, bytesPerSlice), bytesPerSlice)
        remaining = remaining.subarray(bytesPerSlice)
      }
    }

    // Check if slice is complete
    const isSliceComplete = currentSlice.sampleCount >= bytesPerSlice * 0.8 // 80% full

    if (isSliceComplete) {
      this.finalizeCurrentSlice()
    }

    return { slice: currentSlice }
  }

  private getCurrentSlice(): AudioSlice {
    let slice = this.slices.find((s) => s.index === this.currentSliceIndex)

    if (!slice) {
      const bytesPerSlice = this.sliceDurationSec * this.sampleRate * 2 // 2 bytes per sample
      slice = {
        index: this.currentSliceIndex,
        data: new Uint8Array(bytesPerSlice),
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
   * Get audio data for transcription (base64 encoded)
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
   * Force move to the next slice, finalizing the current one regardless of capacity
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
