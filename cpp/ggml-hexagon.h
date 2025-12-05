#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
WSP_GGML_BACKEND_API wsp_ggml_backend_t wsp_ggml_backend_hexagon_init(void);

WSP_GGML_BACKEND_API bool wsp_ggml_backend_is_hexagon(wsp_ggml_backend_t backend);

WSP_GGML_BACKEND_API wsp_ggml_backend_reg_t wsp_ggml_backend_hexagon_reg(void);

#ifdef  __cplusplus
}
#endif
