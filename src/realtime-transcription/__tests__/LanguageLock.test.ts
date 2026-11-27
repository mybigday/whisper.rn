import { LanguageLock } from '../LanguageLock'

describe('LanguageLock', () => {
  it('locks after enough confident detections, ignoring doubtful ones', () => {
    const lock = new LanguageLock({ lockAfter: 3, minProb: 0.5 })
    lock.observe('ja', 0.9)
    lock.observe('en', 0.3) // doubtful: does not count
    lock.observe('ja', 0.8)
    expect(lock.locked).toBeUndefined()
    expect(lock.languageFor(true, 5000)).toBeUndefined()
    lock.observe('ja', 0.7)
    expect(lock.locked).toBe('ja')
    expect(lock.languageFor(false, 1000)).toBe('ja')
  })

  it('re-checks every Nth long final and unlocks on repeated confident disagreement', () => {
    const lock = new LanguageLock({ lockAfter: 1, recheckEvery: 3, recheckMinMs: 3000, unlockAfter: 2, unlockProb: 0.9 })
    lock.observe('en', 0.9)
    const requested = []
    for (let i = 0; i < 6; i += 1) requested.push(lock.languageFor(true, 4000))
    expect(requested).toEqual(['en', 'en', undefined, 'en', 'en', undefined])
    // Short finals are not re-checked
    expect(new LanguageLock({ lockAfter: 1, recheckEvery: 1 }).languageFor(true, 500)).toBeUndefined()

    lock.observe('ja', 0.95)
    expect(lock.locked).toBe('en')
    lock.observe('en', 0.99) // agreement resets the count
    lock.observe('ja', 0.95)
    expect(lock.locked).toBe('en')
    lock.observe('ja', 0.95)
    expect(lock.locked).toBeUndefined()
  })
})
