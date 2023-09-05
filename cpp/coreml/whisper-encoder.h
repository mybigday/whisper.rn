// Wrapper of the Core ML Whisper Encoder model
//
// Code is derived from the work of Github user @wangchou
// ref: https://github.com/wangchou/callCoreMLFromCpp

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_context;

struct whisper_coreml_context * whisper_coreml_init(const char * path_model);
void whisper_coreml_free(struct whisper_coreml_context * ctx);

void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                               float * mel,
                               float * out);

#if __cplusplus
}
#endif
