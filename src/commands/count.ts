import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags } from './flags'

export const countCommand = command(
  {
    name: 'count',
    help: {
      description: 'Display the number of entries in the database'
    },
    flags: {
      ...sharedFlags
    }
  },
  async (argv) => {
    const storePath = argv.flags.storePath
    const engine = new EmbeddingEngine({ storePath, readOnly: true })

    try {
      const count = await engine.count()
      console.log(count)
    } finally {
      await engine.dispose()
    }
  }
)
