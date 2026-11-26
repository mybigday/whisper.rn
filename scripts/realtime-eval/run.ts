/* eslint-disable no-console, no-await-in-loop, import/no-extraneous-dependencies, unicorn/no-process-exit */
// Replays a WAV through RealtimeTranscriber with a host whisper.cpp
// (whisper-server, see README.md) and scores the final transcript.
//
//   npx ts-node --transpile-only scripts/realtime-eval/run.ts \
//     --server <build>/whisper-server --model ggml-base.bin [--dtw base] \
//     (--wav clip.wav [--ref "text"] | --manifest set/manifest.json --item 0 --root <dir>) \
//     [--options '{"vadOptions":{"maxSpeechDurationS":6},"splitOverlapMs":400}'] [--seconds 120]
import { execFileSync, spawn } from 'child_process'
import { readFileSync } from 'fs'
import { dirname, resolve } from 'path'
import { createInterface } from 'readline'
import { RealtimeTranscriber } from '../../src/realtime-transcription/RealtimeTranscriber'
import { StreamingVadSegmenter } from '../../src/realtime-transcription/StreamingVadSegmenter'
import type {
  AudioStreamInterface,
  RealtimeOptions,
  RealtimeTranscribeEvent,
} from '../../src/realtime-transcription/types'
import type { TranscribeOptions, TranscribeResult } from '../../src/index'

const args = new Map<string, string>()
process.argv.slice(2).forEach((arg, i, all) => {
  if (arg.startsWith('--')) args.set(arg.slice(2), all[i + 1] ?? '')
})
const arg = (name: string, fallback?: string) => args.get(name) ?? fallback

/** PCM16 samples of a 16 kHz mono PCM16 WAV */
function readWav(path: string): Int16Array {
  const buf = readFileSync(path)
  let offset = 12
  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4)
    const size = buf.readUInt32LE(offset + 4)
    if (id === 'fmt ') {
      const format = buf.readUInt16LE(offset + 8)
      const channels = buf.readUInt16LE(offset + 10)
      const rate = buf.readUInt32LE(offset + 12)
      const bits = buf.readUInt16LE(offset + 22)
      if (format !== 1 || channels !== 1 || rate !== 16000 || bits !== 16) {
        throw new Error(`${path}: need 16 kHz mono PCM16 (got format ${format}, ${channels} ch, ${rate} Hz, ${bits} bit)`)
      }
    }
    if (id === 'data') {
      const data = buf.subarray(offset + 8, offset + 8 + size)
      return new Int16Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
    }
    offset += 8 + size + (size % 2)
  }
  throw new Error(`${path}: no data chunk`)
}

/** whisper-server as a WhisperContextLike, one request at a time */
async function startServer(server: string, model: string, dtw: string, threads: string) {
  const child = spawn(server, [model, dtw, threads], { stdio: ['pipe', 'pipe', 'inherit'] })
  const lines = createInterface({ input: child.stdout })
  const waiting: Array<(line: string) => void> = []
  lines.on('line', (line) => waiting.shift()?.(line))
  const next = () => new Promise<string>((res) => waiting.push(res))
  const ready = JSON.parse(await next())
  let computeMs = 0
  let requests = 0
  let chain: Promise<unknown> = Promise.resolve()

  const transcribeData = (data: ArrayBuffer, options: TranscribeOptions & { audioCtx?: number | 'auto' }) => {
    const kv: string[] = []
    if (options.language) kv.push(`language=${options.language}`)
    if (options.prompt) kv.push(`prompt_hex=${Buffer.from(options.prompt, 'utf8').toString('hex')}`)
    if (options.audioCtx !== undefined) kv.push(`audio_ctx=${options.audioCtx}`)
    if (options.maxTokens !== undefined) kv.push(`max_tokens=${options.maxTokens}`)
    if (options.singleSegment) kv.push('single_segment=1')
    if (options.temperatureInc !== undefined) kv.push(`temperature_inc=${options.temperatureInc}`)
    if (options.includeTokens) kv.push('include_tokens=1')
    if (options.tokenTimestamps) kv.push('token_timestamps=1')
    const promise = (chain = chain.then(async () => {
      child.stdin.write(`${data.byteLength} ${kv.join(' ')}\n`)
      child.stdin.write(Buffer.from(data))
      const result = JSON.parse(await next())
      computeMs += result.ms
      requests += 1
      return result as TranscribeResult
    })) as Promise<TranscribeResult>
    return { stop: async () => {}, promise }
  }
  return {
    context: { transcribeData },
    dtw: !!ready.dtw,
    stats: () => ({ computeMs, requests }),
    close: () => child.stdin.end(),
  }
}

/**
 * A VAD probability source from vad.cpp probabilities computed up front
 * (vad-probs): the stream is stateful, so they equal what a live stream gives
 */
function precomputedVad(vadProbsBin: string, model: string, wav: string) {
  const { hopSamples, probs } = JSON.parse(
    execFileSync(vadProbsBin, [model, wav], { maxBuffer: 256 * 1024 * 1024 }).toString(),
  ) as { hopSamples: number; probs: number[] }
  return {
    hopSamples,
    sampleRate: 16000,
    createStream: async () => {
      let pending = 0
      let next = 0
      return {
        push: async (audio: Uint8Array) => {
          pending += audio.byteLength / 2
          const n = Math.floor(pending / hopSamples)
          pending -= n * hopSamples
          next += n
          return probs.slice(next - n, next)
        },
        flush: async () => {
          const rest = probs.slice(next)
          next = probs.length
          return rest
        },
        reset: async () => { pending = 0; next = 0 },
        release: async () => {},
      }
    },
  }
}

/** An audio stream that plays the samples in 100 ms chunks as fast as the pipeline takes them */
function replayStream(samples: Int16Array) {
  let onData: (d: any) => void = () => {}
  let onEnd: () => void = () => {}
  let recording = false
  const stream: AudioStreamInterface = {
    initialize: async () => {},
    start: async () => { recording = true },
    stop: async () => { recording = false },
    isRecording: () => recording,
    onData: (cb) => { onData = cb },
    onError: () => {},
    onStatusChange: () => {},
    onEnd: (cb) => { onEnd = cb },
    release: async () => {},
  }
  const play = async () => {
    const chunk = 1600
    for (let i = 0; i < samples.length && recording; i += chunk) {
      const part = samples.slice(i, i + chunk)
      onData({ data: new Uint8Array(part.buffer), sampleRate: 16000, channels: 1, timestamp: Date.now() })
      // Let the segmentation (and queued transcriptions) run between chunks
      await new Promise((res) => setImmediate(res))
    }
    onEnd()
  }
  return { stream, play }
}

const CJK = /[぀-ヿ㐀-䶿一-鿿가-힯]/
const units = (text: string) => {
  const t = text.toLowerCase().replace(/[\p{P}\p{S}]+/gu, ' ')
  return CJK.test(t) ? [...t.replace(/\s+/g, '')] : t.split(/\s+/).filter(Boolean)
}
/** Word (or CJK character) error rate */
function errorRate(ref: string, hyp: string) {
  const r = units(ref)
  const h = units(hyp)
  let prev = Array.from({ length: h.length + 1 }, (_, j) => j)
  for (let i = 1; i <= r.length; i += 1) {
    const cur = [i]
    for (let j = 1; j <= h.length; j += 1) {
      cur[j] = Math.min(prev[j]! + 1, cur[j - 1]! + 1, prev[j - 1]! + (r[i - 1] === h[j - 1] ? 0 : 1))
    }
    prev = cur
  }
  return { rate: r.length > 0 ? prev[h.length]! / r.length : 0, refUnits: r.length }
}

async function main() {
  let samples: Int16Array
  let ref = arg('ref')
  let name = arg('wav') ?? ''
  let wavPath = arg('wav') ?? ''
  if (arg('manifest')) {
    const manifestPath = arg('manifest')!
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))
    const item = manifest.items[Number(arg('item', '0'))]
    const root = arg('root', resolve(dirname(manifestPath), '../../..'))!
    wavPath = resolve(root, item.wav)
    samples = readWav(wavPath)
    const refSrc: string[] = Array.isArray(item.ref_src) ? item.ref_src : JSON.parse(item.ref_src)
    ref = refSrc.join(' ')
    name = item.id
  } else {
    samples = readWav(arg('wav')!)
  }
  const seconds = Number(arg('seconds', '0'))
  if (seconds > 0 && samples.length > seconds * 16000) {
    samples = samples.slice(0, seconds * 16000)
    ref = undefined // a reference for the whole file no longer matches
  }

  const server = await startServer(arg('server')!, arg('model')!, arg('dtw', '-')!, arg('threads', '4')!)
  const options: RealtimeOptions = {
    realtimeProcessingPauseMs: 1e9, // finals only: drafts do not change them
    ...JSON.parse(arg('options', '{}')!),
  }
  if (arg('log')) options.logger = (m) => console.error(m)
  // Offline split rule (cut right at the limit) for comparison
  if (arg('hardcut')) options.probabilitySegmenter = new StreamingVadSegmenter({}, 512, 16000, { splitAtDip: false })
  const finals: RealtimeTranscribeEvent[] = []
  const { stream, play } = replayStream(samples)
  // --vad <silero model> [--vad-probs <bin>]: probabilities of the whole file (not truncated by --seconds)
  const vadContext = arg('vad')
    ? precomputedVad(arg('vad-probs', resolve(dirname(arg('server')!), 'vad-probs'))!, arg('vad')!, wavPath)
    : undefined
  const transcriber = new RealtimeTranscriber(
    { whisperContext: server.context, audioStream: stream, vadContext },
    options,
    {
      onTranscribe: (e) => { if (e.type === 'transcribe' && e.isFinal) finals.push(e) },
      onError: (e) => console.error('error:', e),
    },
  )
  await transcriber.start()
  await play()
  await transcriber.stop()
  server.close()

  finals.sort((a, b) => a.startSample! - b.startSample!)
  const hyp = finals.map((e) => e.data?.result ?? '').join(' ')
  const reasons: Record<string, number> = {}
  finals.forEach((e) => { reasons[e.endReason ?? '?'] = (reasons[e.endReason ?? '?'] ?? 0) + 1 })
  const { computeMs, requests } = server.stats()
  const report: Record<string, unknown> = {
    name,
    audioSec: samples.length / 16000,
    finals: finals.length,
    reasons,
    dtw: server.dtw,
    vad: !!vadContext,
    requests,
    computePerAudio: computeMs / 1000 / (samples.length / 16000),
    options: { ...options, segmenter: options.probabilitySegmenter ? "hardcut" : undefined },
  }
  if (ref) Object.assign(report, { errorRate: errorRate(ref, hyp).rate, refUnits: errorRate(ref, hyp).refUnits })
  if (arg('print') === 'finals') finals.forEach((e) => console.log(`[${(e.startSample! / 16000).toFixed(2)}-${(e.endSample! / 16000).toFixed(2)} ${e.endReason}] ${e.data?.result}`))
  if (arg("print") === "raw") finals.forEach((e) => console.log(JSON.stringify({ s: e.startSample, e: e.endSample, segs: e.data?.segments.map((g: any) => [g.text, (g.tokens ?? []).map((t: any) => `${t.text}@${t.tDtw}/${t.t0}`).join("")]) })))
  console.log(JSON.stringify(report))
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
