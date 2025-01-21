//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include "lib_with_public_kokkos_dependency.h"
#include <iostream>

namespace lib_with_public_kokkos_dependency {

static bool i_initialized_kokkos = false;

void initialize() {
  // if I have to initialize kokkos, I assume I also have to finalize after I
  // did what I needed Kokkos for
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
    i_initialized_kokkos = true;
  }
}

void finalize() {
  if (i_initialized_kokkos and !Kokkos::is_finalized()) {
    Kokkos::finalize();
  }
}

void print(Kokkos::View<int*> a) {
  std::cout << "Hello from lib_with_public_kokkos_dependency\n";
}

}  // namespace lib_with_public_kokkos_dependency
