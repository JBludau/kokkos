// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <impl/Kokkos_Memcmp.hpp>

namespace Kokkos {
namespace Impl {
KOKKOS_FUNCTION int memcmp(void const* lhs, void const* rhs,
                           std::size_t count) {
  auto u1 = static_cast<unsigned char const*>(lhs);
  auto u2 = static_cast<unsigned char const*>(rhs);
  while (count-- != 0) {
    if (*u1 != *u2) {
      return (*u1 < *u2) ? -1 : +1;
    }
    ++u1;
    ++u2;
  }
  return 0;
}
}  // namespace Impl
}  // namespace Kokkos
