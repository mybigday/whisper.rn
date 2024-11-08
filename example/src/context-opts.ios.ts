import { Platform } from 'react-native'

export default {
  useCoreMLIos: true,
  // If you don't want to enable Core ML, you can remove this property
  coreMLModelAsset:
    Platform.OS === 'ios'
      ? {
          filename: 'ggml-base-encoder.mlmodelc',
          assets: [
            require('../assets/ggml-base-encoder.mlmodelc/weights/weight.bin'),
            require('../assets/ggml-base-encoder.mlmodelc/model.mil'),
            require('../assets/ggml-base-encoder.mlmodelc/coremldata.bin'),
          ],
        }
      : undefined,
  useGpu: false, // Enable Metal (Will skip Core ML if enabled)
  useFlashAttn: false,
}
