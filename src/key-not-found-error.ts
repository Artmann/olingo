/**
 * Error thrown when attempting to update a key that does not exist in the database.
 */
export class KeyNotFoundError extends Error {
  constructor(public readonly key: string) {
    super(`Key not found: "${key}"`)
    this.name = 'KeyNotFoundError'
  }
}
