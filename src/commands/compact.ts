import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags, modelFromFlag } from './flags'

export const compactCommand = command(
  {
    name: 'compact',
    help: {
      description: 'Compact the database by removing dead records'
    },
    flags: {
      ...sharedFlags
    }
  },
  async (argv) => {
    const storePath = argv.flags.storePath
    const engine = new EmbeddingEngine({
      storePath,
      model: modelFromFlag(argv.flags.model)
    })

    try {
      const result = await engine.compact()

      console.log(`Records before: ${result.recordsBefore}`)
      console.log(`Records after:  ${result.recordsAfter}`)
      console.log(
        `Size before:    ${(result.bytesBefore / 1024).toFixed(1)} KB`
      )
      console.log(`Size after:     ${(result.bytesAfter / 1024).toFixed(1)} KB`)

      const saved = result.bytesBefore - result.bytesAfter
      if (saved > 0) {
        console.log(`\nSaved ${(saved / 1024).toFixed(1)} KB`)
      } else {
        console.log(`\nNo space reclaimed.`)
      }
    } finally {
      await engine.dispose()
    }
  }
)
