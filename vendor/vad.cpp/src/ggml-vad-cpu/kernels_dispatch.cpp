#include "kernels.h"

namespace vadk_generic { extern const vadk::table table; }
#ifdef VADK_X86_VARIANTS
namespace vadk_avx2   { extern const vadk::table table; }
namespace vadk_avx512 { extern const vadk::table table; }
namespace vadk_avx512vnni { extern const vadk::table table; }
static bool has_avx512() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}
#endif
#ifdef VADK_ARM_VARIANTS
namespace vadk_armv82 { extern const vadk::table table; }
#if defined(__APPLE__)
#include <sys/sysctl.h>
static bool sysctl_flag(const char * name) {
    int v = 0;
    size_t n = sizeof(v);
    return sysctlbyname(name, &v, &n, nullptr, 0) == 0 && v != 0;
}
static bool has_armv82() {
    return sysctl_flag("hw.optional.arm.FEAT_FP16") && sysctl_flag("hw.optional.arm.FEAT_DotProd");
}
#else
#include <sys/auxv.h>
#include <asm/hwcap.h>
static bool has_armv82() {
    const unsigned long hw = getauxval(AT_HWCAP);
    return (hw & HWCAP_ASIMDHP) && (hw & HWCAP_ASIMDDP);
}
#endif
#endif

namespace vadk {

static const table & select() {
#ifdef VADK_X86_VARIANTS
    __builtin_cpu_init();
    if (has_avx512() && __builtin_cpu_supports("avx512vnni")) return vadk_avx512vnni::table;
    if (has_avx512()) return vadk_avx512::table;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return vadk_avx2::table;
#endif
#ifdef VADK_ARM_VARIANTS
    if (has_armv82()) return vadk_armv82::table;
#endif
    return vadk_generic::table;
}

int available(const table ** out, int max) {
    int n = 0;
    auto push = [&](const table & t) { if (n < max) out[n] = &t; ++n; };
    push(vadk_generic::table);
#ifdef VADK_X86_VARIANTS
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) push(vadk_avx2::table);
    if (has_avx512()) push(vadk_avx512::table);
    if (has_avx512() && __builtin_cpu_supports("avx512vnni")) push(vadk_avx512vnni::table);
#endif
#ifdef VADK_ARM_VARIANTS
    if (has_armv82()) push(vadk_armv82::table);
#endif
    return n;
}

const table & get() {
    static const table & t = select();
    return t;
}

} // namespace vadk
