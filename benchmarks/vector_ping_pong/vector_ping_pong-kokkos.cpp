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

#include "Kokkos_Core.hpp"
#include <iostream>

#include <sys/time.h>

template <typename MemorySpacePing, typename MemorySpacePong,
          typename ExecutionSpacePing, typename ExecutionSpacePong,
          typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, bool needs_deep_copy,
          typename VectorValue, typename VectorIndex>
int run_benchmark(VectorValue* ping_data, VectorValue* pong_data,
                  VectorIndex size) {
  int warmup_runs = 10;
  int num_runs    = 100;

  auto warmup_view =
      Kokkos::View<VectorValue*, MemorySpacePing>{"warmup", size};

  auto ping_view = Kokkos::View<VectorValue*, MemorySpacePing>{ping_data, size};
  auto pong_view = Kokkos::View<VectorValue*, MemorySpacePong>{pong_data, size};

  // do warmup with another view so we don't mess up the placement
  for (int i = 0; i < warmup_runs; ++i) {
    Kokkos::parallel_for(
        "warmup inc", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) { ++warmup_view(idx); });
    Kokkos::parallel_for(
        "warmup dec", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) { --warmup_view(idx); });
  }
  Kokkos::fence();

  Kokkos::Timer first_touch_ping{};
  Kokkos::parallel_for(
      "first_touch_ping",
      Kokkos::RangePolicy(ExecutionSpaceFirstTouchPing(), 0, size),
      KOKKOS_LAMBDA(const VectorIndex i) { ping_view(i) = VectorValue{}; });
  Kokkos::fence();
  std::cout << "First touch ping " << first_touch_ping.seconds() << std::endl;

  Kokkos::Timer first_touch_pong;
  Kokkos::parallel_for(
      "first_touch_pong",
      Kokkos::RangePolicy(ExecutionSpaceFirstTouchPong(), 0, size),
      KOKKOS_LAMBDA(const VectorIndex i) { pong_view(i) = VectorValue{}; });
  Kokkos::fence();
  std::cout << "First touch pong " << first_touch_pong.seconds() << std::endl;

  Kokkos::Timer timer;
  for (int i = 0; i < num_runs; ++i) {
    if constexpr (needs_deep_copy)
      Kokkos::deep_copy(ping_view, pong_view);
    else
      Kokkos::fence();
    Kokkos::parallel_for(
        "ping", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) { ++ping_view(idx); });
    if constexpr (needs_deep_copy)
      Kokkos::deep_copy(pong_view, ping_view);
    else
      Kokkos::fence();
    Kokkos::parallel_for(
        "pong", Kokkos::RangePolicy(ExecutionSpacePong(), 0, size),
        KOKKOS_LAMBDA(const VectorIndex idx) { ++pong_view(idx); });
  }
  Kokkos::fence();
  auto totalTime = timer.seconds();
  std::cout << "Elapsed time " << totalTime << " for " << num_runs
            << " runs with vectorlength " << size
            << " resulting average bandwitdh "
            << 1.0e-6 * 2.0 * num_runs * size * (double)sizeof(VectorValue) /
                   totalTime
            << " MB/s" << std::endl;
  // Kokkos::fence();

  // check for errors
  int error_count = 0;
  // since we ended on pong but want to check ping, we need to copy it again.
  Kokkos::deep_copy(ping_view, pong_view);
  Kokkos::parallel_reduce(
      "error_check", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
      KOKKOS_LAMBDA(const VectorIndex i, int& error) {
        error += (ping_view(i) == num_runs * 2) ? 0 : 1;
      },
      error_count);
  Kokkos::fence();
  return error_count;
}

///////////////////////NEW
template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong, typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong>
int benchmark_new_separate_arrays(unsigned size) {
  ValueType* vec_ping = new ValueType[size];
  ValueType* vec_pong = new ValueType[size];

  int rc =
      run_benchmark<Kokkos::SharedSpace, Kokkos::SharedSpace,
                    ExecutionSpacePing, ExecutionSpacePong,
                    ExecutionSpaceFirstTouchPing, ExecutionSpaceFirstTouchPong,
                    true>(vec_ping, vec_pong, size);

  delete[] vec_ping;
  delete[] vec_pong;

  return rc;
}

template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong, typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong>
int benchmark_new_single_array(unsigned size) {
  ValueType* vec_ping_pong = new ValueType[size];

  int rc =
      run_benchmark<Kokkos::SharedSpace, Kokkos::SharedSpace,
                    ExecutionSpacePing, ExecutionSpacePong,
                    ExecutionSpaceFirstTouchPing, ExecutionSpaceFirstTouchPong,
                    false>(vec_ping_pong, vec_ping_pong, size);

  delete[] vec_ping_pong;

  return rc;
}

/////////////////////CUDA_MALLOC_MANAGED
template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong>
int benchmark_cudaMallocManaged_single_array(unsigned size) {
  ValueType* vec_ping_pong;
  cudaMallocManaged(&vec_ping_pong, size * sizeof(ValueType));

  int rc = run_benchmark<
      Kokkos::CudaUVMSpace, Kokkos::HostSpace, Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultHostExecutionSpace, false>(vec_ping_pong, vec_ping_pong,
                                                size);

  cudaFree(vec_ping_pong);

  return rc;
}

template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong>
int benchmark_cudaMallocManaged_separate_arrays(unsigned size) {
  ValueType* vec_ping;
  ValueType* vec_pong;
  cudaMallocManaged(&vec_ping, size * sizeof(ValueType));
  cudaMallocManaged(&vec_pong, size * sizeof(ValueType));

  int rc = run_benchmark<
      Kokkos::CudaUVMSpace, Kokkos::HostSpace, Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultHostExecutionSpace, true>(vec_ping, vec_pong, size);

  cudaFree(vec_ping);
  cudaFree(vec_pong);

  return rc;
}

/////////////////////CUDA_MALLOC
template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong>
int benchmark_cudaMalloc_separate_arrays(unsigned size) {
  ValueType* vec_ping;
  ValueType* vec_pong;
  cudaMalloc(&vec_ping, size * sizeof(ValueType));
  cudaMallocHostPinned(&vec_pong, size * sizeof(ValueType));

  int rc = run_benchmark<
      Kokkos::CudaSpace, Kokkos::HostSpace, Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultHostExecutionSpace, true>(vec_ping, vec_pong, size);

  cudaFree(vec_ping);
  cudaFree(vec_pong);

  return rc;
}

template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong>
int benchmark_cudaMalloc_cudaMallocHostPinned_separate_arrays(unsigned size) {
  ValueType* vec_ping;
  ValueType* vec_pong;
  cudaMalloc(&vec_ping, size * sizeof(ValueType));
  cudaMallocHost(&vec_pong, size * sizeof(ValueType));

  int rc = run_benchmark<
      Kokkos::CudaSpace, Kokkos::HostSpace, Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace,
      Kokkos::DefaultHostExecutionSpace, true>(vec_ping, vec_pong, size);

  cudaFree(vec_ping);
  cudaFree(vec_pong);

  return rc;
}

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  Kokkos::initialize(argc, argv);
  {
    using ValueType = int;
    unsigned size   = 1 << 25;

    ////////////////////////NEW
    int rc = benchmark_new_separate_arrays<ValueType, Kokkos::DefaultExecutionSpace,
                                      Kokkos::DefaultHostExecutionSpace,Kokkos::DefaultExecutionSpace,
                                      Kokkos::DefaultHostExecutionSpace
    >(size);
     if (rc != 0)
     std::cout << "new_separated: error check not successful, "
     "error count:"
     << rc << std::endl;

    rc = benchmark_new_single_array<ValueType,
    Kokkos::DefaultExecutionSpace,
    Kokkos::DefaultHostExecutionSpace,
    Kokkos::DefaultExecutionSpace,
    Kokkos::DefaultHostExecutionSpace
    >(size);

    if (rc != 0)
    std::cout << "new_single: error check not successful, "
    "error count:"
    << rc << std::endl;


    //////////////////////CUDA_MALLOC_MANAGED
    // rc = benchmark_cudaMallocManaged_separate_arrays<
    // ValueType, Kokkos::DefaultExecutionSpace,
    // Kokkos::DefaultHostExecutionSpace>(size);
    // if (rc != 0)
    // std::cout << "cudaMallocManaged_separated: error check not successful, "
    // "error count:"
    // << rc << std::endl;
    // rc = benchmark_cudaMallocManaged_single_array<
    // ValueType, Kokkos::DefaultExecutionSpace,
    // Kokkos::DefaultHostExecutionSpace>(size);
    // if (rc != 0)
    // std::cout << "cudaMallocManaged_single: error check not successful, "
    // "error count:"
    // << rc << std::endl;

    // ////////////////////CUDA_MALLOC
    // rc =
    // benchmark_cudaMalloc_separate_arrays<ValueType,
    // Kokkos::DefaultExecutionSpace,
    // Kokkos::DefaultHostExecutionSpace>(
    // size);
    // if (rc != 0)
    // std::cout << "cudaMalloc_separated: error check not successful, "
    // "error count:"
    // << rc << std::endl;
    // rc = benchmark_cudaMalloc_cudaMallocHostPinned_separate_arrays<
    // ValueType, Kokkos::DefaultExecutionSpace,
    // Kokkos::DefaultHostExecutionSpace>(size);
    // if (rc != 0)
    // std::cout << "cudaMalloc_cudaMallocHostPinned_separated: error check not
    // " "successful, " "error count:"
    // << rc << std::endl;
  }
  Kokkos::finalize();

  return 0;
}
