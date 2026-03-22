/**
 * Error thrown when an embedding's dimension doesn't match the database's expected dimension.
 */
export class DimensionMismatchError extends Error {
  constructor(
    public readonly expectedDimension: number,
    public readonly actualDimension: number
  ) {
    super(
      `Embedding dimension mismatch: expected ${expectedDimension}, got ${actualDimension}`
    )
    this.name = 'DimensionMismatchError'
  }
}
