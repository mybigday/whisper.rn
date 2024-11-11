import { Platform } from 'react-native'

const getCoreMLModelAsset = () =>
  Platform.OS === 'ios'
    ? {
        filename: 'ggml-base-encoder.mlmodelc',
        assets: [
          require('../assets/ggml-base-encoder.mlmodelc/weights/weight.bin'),
          require('../assets/ggml-base-encoder.mlmodelc/model.mil'),
          require('../assets/ggml-base-encoder.mlmodelc/coremldata.bin'),
        ],
      }
    : undefined

export default {
  useGpu: true, // Enable Metal (Will skip Core ML if enabled)
  useFlashAttn: true,

  useCoreMLIos: false,
  coreMLModelAsset: getCoreMLModelAsset(),
}
