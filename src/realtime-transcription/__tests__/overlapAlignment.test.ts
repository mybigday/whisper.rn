import { anchorAfterPrevious, dropRepeatedPrefix, joinUnits, keepUnitsInRange, timedUnits } from '../overlapAlignment'

const token = (text: string, t0: number, tDtw = -1) => ({ id: 0, text, p: 1, plog: 0, t0, t1: -1, tDtw })

describe('overlap alignment', () => {
  it('groups tokens into words and CJK characters, timed in stream samples', () => {
    const units = timedUnits(
      [{ tokens: [token(' Hel', 0), token('lo', 10), token(' world', 20, 25), token('', 40), token('語', 42), token('音', 50)] }],
      16000,
      16000,
    )
    expect(units).toEqual([
      { text: ' Hello', start: 16000 },
      // DTW time preferred over t0
      { text: ' world', start: 16000 + 25 * 160 },
      // The empty token's bytes (and time) go with the character it completes
      { text: '語', start: 16000 + 40 * 160 },
      { text: '音', start: 16000 + 50 * 160 },
    ])
  })

  it('needs tokens with times', () => {
    expect(timedUnits([{}], 0, 16000)).toBeNull()
    expect(timedUnits([{ tokens: [token(' a', -1)] }], 0, 16000)).toBeNull()
  })

  it('keeps the units that start in range, with slack before the start', () => {
    const units = [100, 900, 1000, 1500, 2000].map((start, i) => ({ text: ` u${i}`, start }))
    expect(joinUnits(keepUnitsInRange(units, 1000, 2000, 100))).toBe('u1 u2 u3')
    expect(joinUnits(keepUnitsInRange(units, undefined, 1000, 100))).toBe('u0 u1')
  })

  it('drops a prefix repeating the previous slice', () => {
    const words = (text: string) => text.split(' ').map((t) => ({ text: ` ${t}`, start: 0 }))
    expect(joinUnits(dropRepeatedPrefix(words('ask not what'), words('what your country')))).toBe('your country')
    expect(joinUnits(dropRepeatedPrefix(words('and so, my fellow'), words('My fellow Americans')))).toBe('Americans')
    expect(joinUnits(dropRepeatedPrefix(words('one two'), words('three four')))).toBe('three four')
    // A single repeated CJK character is legitimate
    const chars = (text: string) => [...text].map((t) => ({ text: t, start: 0 }))
    expect(joinUnits(dropRepeatedPrefix(chars('我的'), chars('的話')))).toBe('的話')
    expect(joinUnits(dropRepeatedPrefix(chars('實驗時'), chars('驗時用')))).toBe('用')
  })

  it('anchors a slice right after the previous slice\'s last words', () => {
    const at = (text: string, start: number) => text.split(' ').map((t, i) => ({ text: ` ${t}`, start: start + i * 100 }))
    const previous = at('ask not what your country', 0)
    // Times would put "can" before the cut; the text says it comes after "your country"
    const units = at('your country can do for you', 900)
    expect(anchorAfterPrevious(previous, units, 2000)).toBe(2)
    // Not found past the search window
    expect(anchorAfterPrevious(previous, units, 950)).toBe(-1)
    // A short single word does not anchor
    expect(anchorAfterPrevious(at('so do', 0), at('do it', 0), 2000)).toBe(-1)
  })
})
