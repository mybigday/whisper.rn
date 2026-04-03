/* eslint-disable import/no-extraneous-dependencies */
const path = require('path')
const escape = require('escape-string-regexp')

// eslint-disable-next-line import/no-unresolved
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config')
const pak = require('../package.json')

const root = path.resolve(__dirname, '..')

const modules = Object.keys({
  ...pak.peerDependencies,
})

const metroExclusionList = (patterns, fallbackPattern = /\/__tests__\/.*/) => {
  const escapePattern = (pattern) => {
    if (pattern instanceof RegExp) {
      return pattern.source.replace(/\/|\\\//g, `\\${path.sep}`)
    }

    return pattern
      .replace(/[$()*+.?[\\\]^{|}\-]/g, '\\$&')
      .replaceAll('/', `\\${path.sep}`)
  }

  return new RegExp(
    `(${[...patterns, fallbackPattern].map(escapePattern).join('|')})$`,
  )
}

const config = {
  projectRoot: __dirname,
  watchFolders: [root],

  // We need to make sure that only one version is loaded for peerDependencies
  // So we block them at the root, and alias them to the versions in example's node_modules
  resolver: {
    blockList: metroExclusionList(
      modules.map(
        (m) =>
          new RegExp(`^${escape(path.join(root, 'node_modules', m))}\\/.*$`),
      ),
    ),

    extraNodeModules: modules.reduce((acc, name) => {
      acc[name] = path.join(__dirname, 'node_modules', name)
      return acc
    }, {}),

    assetExts: [
      'bin', // ggml model binary
      'mil', // CoreML model asset
    ],
  },

  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: true,
      },
    }),
  },
}

module.exports = (async () => {
  const defaultConfig = await getDefaultConfig(__dirname)

  return mergeConfig(defaultConfig, {
    ...config,
    resolver: {
      ...config.resolver,
      assetExts: [
        ...defaultConfig.resolver.assetExts,
        ...config.resolver.assetExts,
      ],
    },
  })
})()
