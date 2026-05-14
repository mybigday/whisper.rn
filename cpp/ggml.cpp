#include "ggml-impl.h"

#include <cstdlib>
#include <exception>

static std::terminate_handler previous_terminate_handler;

WSP_GGML_NORETURN static void wsp_ggml_uncaught_exception() {
    wsp_ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool wsp_ggml_uncaught_exception_init = []{
    const char * WSP_GGML_NO_BACKTRACE = getenv("WSP_GGML_NO_BACKTRACE");
    if (WSP_GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    WSP_GGML_ASSERT(prev != wsp_ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(wsp_ggml_uncaught_exception);
    return true;
}();
