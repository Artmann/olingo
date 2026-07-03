/**
 * Error thrown when an embedding's dimension doesn't match the database's expected dimension.
 */
export class DimensionMismatchError extends Error {
  constructor(
    public readonly expectedDimension: number,
    public readonly actualDimension: number
  ) {
    super(
      `Embedding dimension mismatch: expected ${expectedDimension}, got ${actualDimension}. ` +
        `This usually means the database was created with a different embedding model. ` +
        `Open it with the same model, or re-embed the data into a new store.`
    )
    this.name = 'DimensionMismatchError'
  }
}
