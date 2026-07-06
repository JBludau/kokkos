// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MEMCMP_HPP
#define KOKKOS_MEMCMP_HPP

#include <Kokkos_Macros.hpp>

#include <cstddef>

namespace Kokkos {
namespace Impl {
KOKKOS_FUNCTION int memcmp(void const* lhs, void const* rhs, std::size_t count);
}
}  // namespace Kokkos
#endif
