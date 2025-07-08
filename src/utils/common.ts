/**
 * Convert base64 string to Uint8Array
 */
export function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = Buffer.from(base64, 'base64').toString('binary')
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i += 1) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  return bytes
}

/**
 * Convert Uint8Array to base64 string
 */
export function uint8ArrayToBase64(buffer: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < buffer.length; i += 1) {
    binary += String.fromCharCode(buffer[i] || 0) // Handle undefined
  }
  return Buffer.from(binary, 'binary').toString('base64')
}
