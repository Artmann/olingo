import { mkdir } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

/**
 * Ensures the parent directory of `filePath` exists, creating it (and any
 * missing ancestors) if necessary.
 *
 * The path is resolved to an absolute path before calling `mkdir`. This works
 * around a Bun-on-Windows bug where `mkdir('.', { recursive: true })` throws
 * `ENOENT` when `filePath` has no directory component (e.g. a store path of
 * `database.raptor`, whose `dirname` is `.`). Resolving to an absolute path
 * avoids the bare `.` and behaves identically on Node and Bun across platforms.
 */
export async function ensureParentDir(filePath: string): Promise<void> {
  await mkdir(resolve(dirname(filePath)), { recursive: true })
}
