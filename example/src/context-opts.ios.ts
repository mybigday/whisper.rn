import { Platform } from 'react-native'

export default {
  useCoreMLIos: true,
  // If you don't want to enable Core ML, you can remove this property
  coreMLModelAsset:
    Platform.OS === 'ios'
      ? {
          filename: 'ggml-tiny.en-encoder.mlmodelc',
          assets: [
            require('../assets/ggml-tiny.en-encoder.mlmodelc/weights/weight.bin'),
            require('../assets/ggml-tiny.en-encoder.mlmodelc/model.mil'),
            require('../assets/ggml-tiny.en-encoder.mlmodelc/coremldata.bin'),
          ],
        }
      : undefined,
  useGpu: false, // Enable Metal (Will skip Core ML if enabled)
}
