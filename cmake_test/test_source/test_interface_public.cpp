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

#include <lib_without_kokkos_dependency.h>
#include <lib_with_interface_dependency_on_lib_with_public_kokkos_dependency.h>

#include <cstdio>
#include <iostream>

extern "C" void print_fortran_();
void print_plain_cxx();

int main(int argc, char* argv[]) {
  lib_without_kokkos_dependency::print();
  Kokkos::initialize(argc, argv);
  {
    print_fortran_();
    print_plain_cxx();
    lib_with_interface_dependency_on_lib_with_public_kokkos_dependency::print(
        lib_with_public_kokkos_dependency::
            StructOfLibWithPublicKokkosDependency{});
  }
  Kokkos::finalize();
}
