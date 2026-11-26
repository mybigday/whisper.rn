import type { TranscribeToken } from '../NativeRNWhisper'

/** A word (spaced scripts) or a single character token (CJK, kana) with its start time */
export type TimedUnit = {
  text: string
  /** Start in stream samples, -1 if unknown */
  start: number
}

// Han, Hiragana, Katakana, Hangul syllables, CJK punctuation and full-width forms
const CJK = /[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]/

/**
 * Units of a decode from its tokens, timed in stream samples. Word times are
 * the first token's DTW time (`tDtw`), else its token timestamp (`t0`).
 * A token that starts with a space starts a word; a CJK token is a unit of
 * its own. Tokens with empty text (bytes of a character completed by the next
 * token) pass their time on to it.
 */
export function timedUnits(
  segments: Array<{ tokens?: TranscribeToken[] }>,
  sliceStartSample: number,
  sampleRate: number,
): TimedUnit[] | null {
  const units: TimedUnit[] = []
  let carriedStart = -1
  let lastWasCjk = false
  let anyTime = false
  // eslint-disable-next-line no-restricted-syntax
  for (const segment of segments) {
    if (!segment.tokens) return null
    // eslint-disable-next-line no-restricted-syntax
    for (const token of segment.tokens) {
      const t = token.tDtw >= 0 ? token.tDtw : token.t0
      // Token times are in 10 ms units
      const time = t >= 0 ? sliceStartSample + Math.round((t * sampleRate) / 100) : -1
      if (time >= 0) anyTime = true
      if (token.text === '') {
        if (carriedStart < 0) carriedStart = time
      } else {
        const start = carriedStart >= 0 ? carriedStart : time
        carriedStart = -1
        const isCjk = CJK.test(token.text)
        const startsWord = /^\s/.test(token.text) || isCjk || lastWasCjk || units.length === 0
        if (startsWord) {
          units.push({ text: token.text, start })
        } else {
          units[units.length - 1]!.text += token.text
        }
        lastWasCjk = isCjk
      }
    }
  }
  return anyTime ? units : null
}

const normalize = (text: string) =>
  text.trim().toLowerCase().replace(/[\p{P}\p{S}]+/gu, '')

/**
 * Drop units that repeat the end of `previous` (the previous slice's kept
 * units) at the start of `units`, up to 5 units. A single CJK character
 * repeats legitimately, so at least two are needed there.
 */
export function dropRepeatedPrefix(previous: TimedUnit[], units: TimedUnit[]): TimedUnit[] {
  const prev = previous.map((u) => normalize(u.text)).filter(Boolean)
  const cur = units.map((u) => normalize(u.text))
  for (let k = Math.min(5, prev.length, cur.length); k >= 1; k -= 1) {
    const tail = prev.slice(prev.length - k)
    const head = cur.slice(0, k)
    if (tail.every((w, i) => w === head[i])) {
      const singleCjk = k === 1 && CJK.test(units[0]!.text)
      if (!singleCjk) return units.slice(k)
    }
  }
  return units
}

/**
 * The units of a slice decoded with overlap audio around a split: those that
 * start in [keepFrom, keepUntil). A unit starting up to `slackSamples` before
 * keepFrom is kept too (decode times are approximate); the repeated-prefix
 * check against the previous slice removes what was already kept there.
 */
export function keepUnitsInRange(
  units: TimedUnit[],
  keepFrom: number | undefined,
  keepUntil: number | undefined,
  slackSamples: number,
): TimedUnit[] {
  return units.filter((u) => {
    if (u.start < 0) return keepUntil === undefined
    if (keepFrom !== undefined && u.start < keepFrom - slackSamples) return false
    if (keepUntil !== undefined && u.start >= keepUntil) return false
    return true
  })
}

export const joinUnits = (units: TimedUnit[]) => units.map((u) => u.text).join('').trim()

/**
 * Where a slice that starts with overlap audio should begin, by text: right
 * after the previous slice's last kept units (2 or 3 of them, or a single
 * spaced word), found among the units starting before `searchUntil`. DTW times
 * are compressed at the end of a decode and early at its start, so a word just
 * after the cut can look like it is before it, in both slices; the text match
 * keeps it. Returns -1 when there is no match.
 */
export function anchorAfterPrevious(
  previous: TimedUnit[],
  units: TimedUnit[],
  searchUntil: number,
): number {
  const prev = previous.map((u) => normalize(u.text)).filter(Boolean)
  const cur = units.map((u) => normalize(u.text))
  const searchable = units.findIndex((u) => u.start >= searchUntil)
  const limit = searchable < 0 ? units.length : searchable
  for (let k = Math.min(3, prev.length); k >= 1; k -= 1) {
    const tail = prev.slice(prev.length - k)
    // A single unit only anchors as a spaced word of some length
    if (k === 1 && (CJK.test(tail[0]!) || tail[0]!.length < 4)) break
    for (let end = k; end <= limit; end += 1) {
      if (tail.every((w, i) => w === cur[end - k + i])) return end
    }
  }
  return -1
}
