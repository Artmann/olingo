import { open } from 'node:fs/promises'
import { deserializeDataRecord, deserializeHeader } from './data-format'
import { headerSize, recordMagic } from './constants'
import type { VerifyResult, VerifyIssue } from './types'

/**
 * Verify the integrity of a database file by scanning all records
 * and validating magic numbers, checksums, and structure.
 */
export async function verifyDatabase(dataPath: string): Promise<VerifyResult> {
  let fileHandle
  try {
    fileHandle = await open(dataPath, 'r')
  } catch {
    // File doesn't exist
    return {
      totalRecords: 0,
      validRecords: 0,
      corruptRecords: 0,
      issues: []
    }
  }

  try {
    const fileStat = await fileHandle.stat()
    const fileSize = fileStat.size

    if (fileSize < headerSize) {
      return {
        totalRecords: 0,
        validRecords: 0,
        corruptRecords: 0,
        issues:
          fileSize > 0
            ? [
                {
                  offset: 0,
                  message: 'File too small to contain a valid header'
                }
              ]
            : []
      }
    }

    // Read and validate header
    const headerBuffer = new Uint8Array(headerSize)
    await fileHandle.read(headerBuffer, 0, headerSize, 0)
    const header = deserializeHeader(headerBuffer)

    if (!header) {
      return {
        totalRecords: 0,
        validRecords: 0,
        corruptRecords: 0,
        issues: [{ offset: 0, message: 'Invalid file header' }]
      }
    }

    // Scan records
    let offset = headerSize
    let totalRecords = 0
    let validRecords = 0
    let corruptRecords = 0
    const issues: VerifyIssue[] = []

    // Read the entire file into memory for scanning
    const fileData = new Uint8Array(fileSize)
    await fileHandle.read(fileData, 0, fileSize, 0)

    while (offset < fileSize) {
      totalRecords++

      const result = deserializeDataRecord(fileData, offset)
      if (result === null) {
        corruptRecords++
        issues.push({
          offset,
          message: `Corrupt or unreadable record at byte offset ${offset}`
        })
        // Try to skip forward to find next record
        // We don't know the record size, so advance by 1 byte and scan for magic
        offset++
        // Scan for next record magic or end of file
        while (offset < fileSize - 4) {
          const view = new DataView(
            fileData.buffer,
            fileData.byteOffset + offset
          )
          const magic = view.getUint32(0, true)
          if (magic === recordMagic) {
            break
          }
          offset++
        }
        if (offset >= fileSize - 4) {
          break
        }
      } else {
        validRecords++
        offset += result.bytesRead
      }
    }

    return {
      totalRecords,
      validRecords,
      corruptRecords,
      issues
    }
  } finally {
    await fileHandle.close()
  }
}
