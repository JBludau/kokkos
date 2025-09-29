// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HIP_ERROR_HPP
#define KOKKOS_HIP_ERROR_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>

#include <hip/hip_runtime.h>

namespace Kokkos {
namespace Impl {

void hip_internal_error_throw(hipError_t e, const char* name,
                              const char* file = nullptr, const int line = 0);

inline void hip_internal_safe_call(hipError_t e, const char* name,
                                   const char* file = nullptr,
                                   const int line   = 0) {
  if (hipSuccess != e) {
    hip_internal_error_throw(e, name, file, line);
  }
}

}  // namespace Impl
}  // namespace Kokkos

#define KOKKOS_IMPL_HIP_SAFE_CALL(call) \
  Kokkos::Impl::hip_internal_safe_call(call, #call, __FILE__, __LINE__)

#endif
