#!/usr/bin/env node

// Release workflow helper: fills native-artifacts.json with the SHA-256 of each
// archive (expected at the package root) right before `npm publish`, so the
// published package's postinstall can verify what it downloads.

const crypto = require('crypto')
const fs = require('fs')
const path = require('path')

const installDir = __dirname
const packageRoot = path.resolve(installDir, '..')
const manifestPath = path.join(installDir, 'native-artifacts.json')

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))

function sha256File(filePath) {
  const hash = crypto.createHash('sha256')
  hash.update(fs.readFileSync(filePath))
  return hash.digest('hex')
}

manifest.artifacts.forEach((artifact) => {
  const archivePath = path.join(packageRoot, artifact.assetName)

  if (!fs.existsSync(archivePath)) {
    throw new Error(`Missing native artifact archive: ${archivePath}`)
  }

  artifact.sha256 = sha256File(archivePath)
  console.log(`${artifact.assetName}: ${artifact.sha256}`)
})

fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
