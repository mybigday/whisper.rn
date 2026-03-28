
import { RingBuffer } from '../RingBuffer'

describe('RingBuffer', () => {
    it('should initialize with correct capacity', () => {
        const buffer = new RingBuffer(10)
        expect(buffer.getCapacity()).toBe(10)
        expect(buffer.getLength()).toBe(0)
        expect(buffer.isEmpty()).toBe(true)
        expect(buffer.isFull()).toBe(false)
    })

    it('should write and read data correctly', () => {
        const buffer = new RingBuffer(10)
        const data = new Uint8Array([1, 2, 3])

        buffer.write(data)
        expect(buffer.getLength()).toBe(3)

        const read = buffer.read()
        expect(read.length).toBe(3)
        expect(read[0]).toBe(1)
        expect(read[1]).toBe(2)
        expect(read[2]).toBe(3)
    })

    it('should handle wrapping around', () => {
        const buffer = new RingBuffer(5)

        // Fill buffer
        buffer.write(new Uint8Array([1, 2, 3]))
        expect(buffer.getLength()).toBe(3)

        // Add more to cause wrap
        buffer.write(new Uint8Array([4, 5, 6]))

        // Capacity is 5. We wrote 3, then 3. Total 6.
        // Buffer should contain [2, 3, 4, 5, 6] (oldest '1' overwritten)
        expect(buffer.getLength()).toBe(5)
        expect(buffer.isFull()).toBe(true)

        const read = buffer.read()
        expect(read.length).toBe(5)
        expect(read[0]).toBe(2)
        expect(read[1]).toBe(3)
        expect(read[2]).toBe(4)
        expect(read[3]).toBe(5)
        expect(read[4]).toBe(6)
    })

    it('should handle writing larger than capacity', () => {
        const buffer = new RingBuffer(5)
        // Write 7 bytes
        buffer.write(new Uint8Array([1, 2, 3, 4, 5, 6, 7]))

        expect(buffer.getLength()).toBe(5)
        const read = buffer.read()
        // Should keep last 5: [3, 4, 5, 6, 7]
        expect(read[0]).toBe(3)
        expect(read[4]).toBe(7)
    })

    it('should clear buffer', () => {
        const buffer = new RingBuffer(5)
        buffer.write(new Uint8Array([1, 2, 3]))
        buffer.clear()

        expect(buffer.getLength()).toBe(0)
        expect(buffer.read().length).toBe(0)
    })
})
