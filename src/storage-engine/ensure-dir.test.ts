import { describe, it, expect, afterEach } from 'vitest'
import { existsSync } from 'node:fs'
import { rm } from 'node:fs/promises'
import { ensureParentDir } from './ensure-dir'

describe('ensureParentDir', () => {
  afterEach(async () => {
    try {
      await rm('./tmp-ensure-dir', { recursive: true, force: true })
    } catch {
      // ignore
    }
  })

  it('does not throw for a bare filename (parent dir is ".")', async () => {
    // Regression: on Bun/Windows mkdir('.', { recursive: true }) throws ENOENT.
    await expect(ensureParentDir('database.raptor')).resolves.toBeUndefined()
  })

  it('does not throw for a "./"-prefixed filename', async () => {
    await expect(ensureParentDir('./database.raptor')).resolves.toBeUndefined()
  })

  it('creates missing nested parent directories', async () => {
    await ensureParentDir('./tmp-ensure-dir/a/b/file.raptor')
    expect(existsSync('./tmp-ensure-dir/a/b')).toBe(true)
  })
})
