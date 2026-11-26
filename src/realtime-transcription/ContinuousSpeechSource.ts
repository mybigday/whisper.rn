import type { StreamVadContextLike, VadStreamLike } from './StreamVad'

/**
 * A probability source that calls everything speech: `RealtimeTranscriber`
 * uses it without a VAD, so the segmenter only splits at the maximum segment
 * duration (at the quietest moment it finds, see `StreamingVadSegmenter`).
 */
export class ContinuousSpeechSource implements StreamVadContextLike {
  readonly hopSamples: number

  readonly sampleRate: number

  constructor(sampleRate = 16000, hopSamples = 512) {
    this.sampleRate = sampleRate
    this.hopSamples = hopSamples
  }

  async createStream(): Promise<VadStreamLike> {
    const { hopSamples } = this
    let pending = 0
    return {
      async push(audio: Uint8Array) {
        pending += audio.byteLength / 2
        const hops = Math.floor(pending / hopSamples)
        pending -= hops * hopSamples
        return new Array(hops).fill(1)
      },
      async flush() {
        const probs = pending > 0 ? [1] : []
        pending = 0
        return probs
      },
      async reset() {
        pending = 0
      },
      async release() {},
    }
  }
}
