import { describe, it, expect, afterEach } from 'vitest'
import { mkdir, writeFile, rm } from 'node:fs/promises'
import { FileLock } from './file-lock'

const testLockPath = './test-data/stale-lock-test.lock'

describe('Stale Lock Detection', () => {
  afterEach(async () => {
    try {
      await rm('./test-data', { recursive: true, force: true })
    } catch {
      // ignore
    }
  })

  it('should detect and recover from stale lock', async () => {
    // Create a lock file with a non-existent PID
    await mkdir('./test-data', { recursive: true })
    // PID 999999999 is extremely unlikely to exist
    await writeFile(testLockPath, '999999999\n')

    const lock = new FileLock(testLockPath, 5000)

    // Should acquire despite existing lock file (stale PID)
    await lock.acquire()
    expect(lock.isLocked()).toBe(true)
    await lock.release()
  })

  it('should not remove lock from live process', async () => {
    // Create a lock file with our own PID (definitely alive)
    await mkdir('./test-data', { recursive: true })
    await writeFile(testLockPath, `${process.pid}\n`)

    const lock = new FileLock(testLockPath, 500) // Short timeout

    // Should fail because the PID is alive
    await expect(lock.acquire()).rejects.toThrow()
  })

  it('should handle empty lock file as stale', async () => {
    await mkdir('./test-data', { recursive: true })
    await writeFile(testLockPath, '')

    const lock = new FileLock(testLockPath, 5000)
    await lock.acquire()
    expect(lock.isLocked()).toBe(true)
    await lock.release()
  })

  it('should handle corrupt lock file as stale', async () => {
    await mkdir('./test-data', { recursive: true })
    await writeFile(testLockPath, 'not-a-pid')

    const lock = new FileLock(testLockPath, 5000)
    await lock.acquire()
    expect(lock.isLocked()).toBe(true)
    await lock.release()
  })
})
