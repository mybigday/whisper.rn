# Tips & Tricks

This document is a collection of tips and tricks for using whisper.rn. If you have any suggestions, feel free to open a discussion or submit a Pull Request.

## Choose model type

Refer to the [Memory Usage](https://github.com/ggerganov/whisper.cpp#memory-usage) section in whisper.cpp for further information.

To achieve the best possible inference quality, you should choose an appropriate model based on the type of device and performance evaluation.

For instance, you can utilize libraries like [react-native-device-info](https://github.com/react-native-device-info/react-native-device-info) or [react-native-vitals](https://github.com/robinpowered/react-native-vitals) (currently unmaintained) to detect the device type and memory usage.

## Use a quantized model

Using a [quantized model](https://github.com/ggerganov/whisper.cpp#quantization) can decrease memory usage and disk space, albeit potentially at the cost of reduced accuracy.

It's worth noting that the q8 model demonstrated performance improvements in our Android tests (on devices using Qualcomm or Google SoCs).

## Change max threads in TranscribeOptions

The default maxThreads value of TranscribeOptions is `2 for 4-core devices, 4 for more cores`.

This is the optimal configuration based on our tests across numerous mobile devices. However, it may not apply universally. If you wish to change it, we advise against using all cores or fewer than 2.
