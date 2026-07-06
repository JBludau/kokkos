// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MEMCMP_HPP
#define KOKKOS_MEMCMP_HPP

#include <Kokkos_Macros.hpp>

#include <cstddef>

namespace Kokkos {
namespace Impl {
KOKKOS_INLINE_FUNCTION int memcmp(void const* lhs, void const* rhs,
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
#endif
