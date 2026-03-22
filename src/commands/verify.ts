import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags } from './flags'

export const verifyCommand = command(
  {
    name: 'verify',
    help: {
      description: 'Verify the integrity of a database file'
    },
    flags: {
      ...sharedFlags
    }
  },
  async (argv) => {
    const storePath = argv.flags.storePath
    const engine = new EmbeddingEngine({ storePath, readOnly: true })

    try {
      const result = await engine.verify()

      console.log(`Total records:   ${result.totalRecords}`)
      console.log(`Valid records:   ${result.validRecords}`)
      console.log(`Corrupt records: ${result.corruptRecords}`)

      if (result.issues.length > 0) {
        console.log(`\nIssues found:`)
        for (const issue of result.issues) {
          console.log(`  - Offset ${issue.offset}: ${issue.message}`)
        }
        process.exitCode = 1
      } else {
        console.log(`\nDatabase is healthy.`)
      }
    } finally {
      await engine.dispose()
    }
  }
)
