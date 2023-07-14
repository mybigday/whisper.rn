// eslint-disable-next-line import/no-extraneous-dependencies
exports.extendAssetExts = (assetExts = require('metro-config/src/defaults/defaults').assetExts) => [
  ...assetExts,
  'bin', // ggml model binary
  'mil', // CoreML model asset
]
