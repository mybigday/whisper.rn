#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void wsp_ggml_critical_section_start(void);
void wsp_ggml_critical_section_end(void);

#ifdef __cplusplus
}
#endif
