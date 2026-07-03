import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags, modelFromFlag } from './flags'

export const keysCommand = command(
  {
    name: 'keys',
    help: {
      description: 'List all keys in the database'
    },
    flags: {
      ...sharedFlags
    }
  },
  async (argv) => {
    const storePath = argv.flags.storePath
    const engine = new EmbeddingEngine({
      storePath,
      readOnly: true,
      model: modelFromFlag(argv.flags.model)
    })

    try {
      const keys = await engine.keys()
      for (const key of keys) {
        console.log(key)
      }
    } finally {
      await engine.dispose()
    }
  }
)
