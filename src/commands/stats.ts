import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags } from './flags'

export const statsCommand = command(
  {
    name: 'stats',
    help: {
      description: 'Display database statistics'
    },
    flags: {
      ...sharedFlags
    }
  },
  async (argv) => {
    const storePath = argv.flags.storePath
    const engine = new EmbeddingEngine({ storePath, readOnly: true })

    try {
      const s = await engine.stats()
      console.log(`Records:       ${s.recordCount}`)
      console.log(`Data file:     ${(s.dataFileSize / 1024).toFixed(1)} KB`)
      console.log(`WAL file:      ${(s.walFileSize / 1024).toFixed(1)} KB`)
      console.log(`Dimension:     ${s.dimension}`)
      console.log(`Read-only:     ${s.isReadOnly}`)
    } finally {
      await engine.dispose()
    }
  }
)
