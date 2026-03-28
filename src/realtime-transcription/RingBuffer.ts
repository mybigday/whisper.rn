/**
 * RingBuffer - A fixed-size circular buffer for audio data
 *
 * This class implements a ring buffer (circular buffer) that maintains
 * a fixed memory footprint regardless of how much data is written.
 * It's designed for pre-recording audio where we only need to keep
 * the last N seconds of audio before speech is detected.
 *
 * Key features:
 * - Fixed memory allocation (no unbounded growth)
 * - O(1) write operations
 * - Preserves most recent data when buffer is full
 */
export class RingBuffer {
    private buffer: Uint8Array

    private writeIndex = 0

    private dataLength = 0

    private readonly maxBytes: number

    /**
     * Create a new RingBuffer
     * @param maxBytes Maximum buffer size in bytes
     */
    constructor(maxBytes: number) {
        this.maxBytes = maxBytes
        this.buffer = new Uint8Array(maxBytes)
    }

    /**
     * Write audio data to the buffer
     * If data exceeds buffer capacity, oldest data is overwritten
     * @param data Audio data to write
     */
    write(data: Uint8Array): void {
        if (data.length === 0) return

        if (data.length >= this.maxBytes) {
            // Data is larger than buffer, only keep the last maxBytes
            const startOffset = data.length - this.maxBytes
            this.buffer.set(data.subarray(startOffset))
            this.writeIndex = 0
            this.dataLength = this.maxBytes
            return
        }

        const spaceToEnd = this.maxBytes - this.writeIndex

        if (data.length <= spaceToEnd) {
            // Data fits without wrapping
            this.buffer.set(data, this.writeIndex)
            this.writeIndex += data.length
            if (this.writeIndex >= this.maxBytes) {
                this.writeIndex = 0
            }
        } else {
            // Data needs to wrap around
            this.buffer.set(data.subarray(0, spaceToEnd), this.writeIndex)
            const remaining = data.length - spaceToEnd
            this.buffer.set(data.subarray(spaceToEnd), 0)
            this.writeIndex = remaining
        }

        // Update data length (capped at maxBytes)
        this.dataLength = Math.min(this.dataLength + data.length, this.maxBytes)
    }

    /**
     * Read all available data from the buffer in correct order
     * Does NOT clear the buffer (use clear() for that)
     * @returns Uint8Array containing the buffered data in order
     */
    read(): Uint8Array {
        if (this.dataLength === 0) {
            return new Uint8Array(0)
        }

        const result = new Uint8Array(this.dataLength)

        if (this.dataLength < this.maxBytes) {
            // Buffer hasn't wrapped yet, data starts at 0
            result.set(this.buffer.subarray(0, this.dataLength))
        } else {
            // Buffer has wrapped, need to reconstruct order
            const readStart = this.writeIndex
            const firstPartLength = this.maxBytes - readStart

            // Copy from readStart to end
            result.set(this.buffer.subarray(readStart, this.maxBytes), 0)
            // Copy from start to writeIndex
            result.set(this.buffer.subarray(0, this.writeIndex), firstPartLength)
        }

        return result
    }

    /**
     * Clear the buffer and reset indices
     */
    clear(): void {
        this.writeIndex = 0
        this.dataLength = 0
        // Note: We don't need to zero the buffer, just reset indices
    }

    /**
     * Get the current amount of data in the buffer
     * @returns Number of bytes currently in the buffer
     */
    getLength(): number {
        return this.dataLength
    }

    /**
     * Get the maximum capacity of the buffer
     * @returns Maximum buffer size in bytes
     */
    getCapacity(): number {
        return this.maxBytes
    }

    /**
     * Check if the buffer is empty
     * @returns true if buffer contains no data
     */
    isEmpty(): boolean {
        return this.dataLength === 0
    }

    /**
     * Check if the buffer is full
     * @returns true if buffer has reached capacity
     */
    isFull(): boolean {
        return this.dataLength >= this.maxBytes
    }

    /**
     * Get the fill percentage of the buffer
     * @returns Number between 0 and 1 representing how full the buffer is
     */
    getFillRatio(): number {
        return this.dataLength / this.maxBytes
    }
}
