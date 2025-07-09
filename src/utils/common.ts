const Buffer: any = global.Buffer || require('safe-buffer').Buffer

/**
 * Convert base64 string to Uint8Array
 */
export function base64ToUint8Array(base64: string): Uint8Array {
  const buffer = Buffer.from(base64, 'base64')
  return new Uint8Array(buffer)
}

/**
 * Convert Uint8Array to base64 string
 */
export function uint8ArrayToBase64(buffer: Uint8Array): string {
  const buf = Buffer.from(buffer)
  return buf.toString('base64')
}
