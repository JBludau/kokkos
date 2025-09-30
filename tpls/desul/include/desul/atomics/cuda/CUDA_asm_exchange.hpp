#include <limits>

namespace desul {
namespace Impl {

#include <desul/atomics/cuda/cuda_cc7_asm_exchange.inc>

#ifndef DESUL_CUDA_ARCH_IS_PRE_HOPPER && defined DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
#include <desul/atomics/cuda/cuda_cc9_asm_exchange.inc>
#endif
}  // namespace Impl
}  // namespace desul
