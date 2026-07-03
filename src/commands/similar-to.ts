import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { KeyNotFoundError } from '../key-not-found-error'
import { sharedFlags, searchFlags, modelFromFlag } from './flags'

export const similarToCommand = command(
  {
    name: 'similar-to',
    parameters: ['<key>'],
    flags: {
      ...sharedFlags,
      ...searchFlags
    },
    help: {
      description: 'Find the documents most similar to an existing key',
      examples: [
        'olingo similar-to doc1',
        'olingo similar-to -l 5 -m 0.7 doc1',
        'olingo similar-to -s ./my-db.raptor doc1'
      ]
    }
  },
  async (argv) => {
    const engine = new EmbeddingEngine({
      storePath: argv.flags.storePath,
      model: modelFromFlag(argv.flags.model)
    })

    try {
      const [key] = argv._
      const results = await engine.similarTo(
        key,
        argv.flags.limit,
        argv.flags.minSimilarity
      )

      if (results.length === 0) {
        console.log('No results found')
      } else {
        console.log(`Found ${results.length} result(s):\n`)
        for (const result of results) {
          console.log(`${result.key}: ${result.similarity.toFixed(6)}`)
        }
      }
    } catch (error) {
      if (error instanceof KeyNotFoundError) {
        console.log(`Key "${argv._[0]}" not found`)
        process.exit(1)
      }
      throw error
    } finally {
      await engine.dispose()
    }
  }
)
