import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 30000, // 30s to allow for model download on first run
    // Test files share the ./test-data directory and each cleans it up in
    // afterEach (rm -rf ./test-data). Running files in parallel lets one file's
    // cleanup delete the directory out from under another file mid-test,
    // causing flaky ENOENT / empty-read failures. Run files serially to isolate.
    fileParallelism: false
  }
})
