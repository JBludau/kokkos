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
#include "Kokkos_Macros.hpp"

#include <iostream>
#include <fstream>

#include <sys/time.h>

template <typename MemorySpacePing, typename MemorySpacePong,
          typename ExecutionSpacePing, typename ExecutionSpacePong,
          typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, bool needs_deep_copy,
          typename VectorValue, typename VectorIndex>
std::tuple<int, double> run_benchmark(VectorValue* ping_data,
                                      VectorValue* pong_data, VectorIndex size,
                                      unsigned warmup_runs,
                                      unsigned num_pingpongs) {
  auto warmup_view =
      Kokkos::View<VectorValue*, MemorySpacePing>{"warmup", size};

  auto ping_view = Kokkos::View<VectorValue*, MemorySpacePing>{ping_data, size};
  auto pong_view = Kokkos::View<VectorValue*, MemorySpacePong>{pong_data, size};

  // do warmup with another view so we don't mess up the placement
  for (unsigned i = 0; i < warmup_runs; ++i) {
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
      KOKKOS_LAMBDA(const VectorIndex idx) { ping_view(idx) = VectorValue{}; });
  Kokkos::fence();
  // std::cout << "First touch ping " << first_touch_ping.seconds() <<
  // std::endl;

  Kokkos::Timer first_touch_pong;
  Kokkos::parallel_for(
      "first_touch_pong",
      Kokkos::RangePolicy(ExecutionSpaceFirstTouchPong(), 0, size),
      KOKKOS_LAMBDA(const VectorIndex idx) { pong_view(idx) = VectorValue{}; });
  Kokkos::fence();
  // std::cout << "First touch pong " << first_touch_pong.seconds() <<
  // std::endl;

  Kokkos::Timer timer;
  for (unsigned i = 0; i < num_pingpongs; ++i) {
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

  // check for errors
  int error_count = 0;
  // since we ended on pong but want to check ping, we need to copy it again.
  Kokkos::deep_copy(ping_view, pong_view);
  Kokkos::parallel_reduce(
      "error_check", Kokkos::RangePolicy(ExecutionSpacePing(), 0, size),
      KOKKOS_LAMBDA(const VectorIndex i, int& error) {
        error += (ping_view(i) == static_cast<VectorValue>(num_pingpongs) * 2)
                     ? 0
                     : 1;
      },
      error_count);
  Kokkos::fence();
  return std::make_tuple(error_count, totalTime);
}

//// ALLOCATOR DEALLOCATOR
struct ManagedMalloc {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    T* ptr;
#ifdef KOKKOS_ENABLE_CUDA
    cudaMallocManaged(&ptr, size * sizeof(T));
#elif defined KOKKOS_ENABLE_HIP
    hipMallocManaged(&ptr, size * sizeof(T));
#endif
    return ptr;
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
#ifdef KOKKOS_ENABLE_CUDA
    cudaFree(ptr);
#elif defined KOKKOS_ENABLE_HIP
    hipFree(ptr);
#endif
  }
};

struct HostPinnedMalloc {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    T* ptr;
#ifdef KOKKOS_ENABLE_CUDA
    cudaMallocHost(&ptr, size * sizeof(T));
#elif defined KOKKOS_ENABLE_HIP
    hipHostMalloc(&ptr, size * sizeof(T));
#endif
    return ptr;
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
#ifdef KOKKOS_ENABLE_CUDA
    cudaFree(ptr);
#elif defined KOKKOS_ENABLE_HIP
    hipFree(ptr);
#endif
  }
};

struct DeviceMalloc {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    T* ptr;
#ifdef KOKKOS_ENABLE_CUDA
    cudaMalloc(&ptr, size * sizeof(T));
#elif defined KOKKOS_ENABLE_HIP
    hipMalloc(&ptr, size * sizeof(T));
#endif
    return ptr;
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
#ifdef KOKKOS_ENABLE_CUDA
    cudaFree(ptr);
#elif defined KOKKOS_ENABLE_HIP
    hipFree(ptr);
#endif
  }
};

struct StdMalloc {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    return std::malloc(size * sizeof(T));
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
    std::free(ptr);
  }
};

struct StdNew {
  template <typename T>
  static constexpr T* allocate(size_t size) {
    return new T[size];
  }

  template <typename T>
  static constexpr void deallocate(T* ptr) {
    delete[] ptr;
  }
};

struct NONE {};

///////////////////////single and double array
template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong, typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, typename AllocatorPing,
          typename AllocatorPong, typename IndexType>
auto benchmark_views(IndexType size, int warmups, int pingpongs, AllocatorPing,
                     AllocatorPong) {
  ValueType* vec_ping = AllocatorPing::template allocate<ValueType>(size);
  ValueType* vec_pong = AllocatorPong::template allocate<ValueType>(size);

  auto rc =
      run_benchmark<Kokkos::SharedSpace, Kokkos::SharedSpace,
                    ExecutionSpacePing, ExecutionSpacePong,
                    ExecutionSpaceFirstTouchPing, ExecutionSpaceFirstTouchPong,
                    true>(vec_ping, vec_pong, size, warmups, pingpongs);

  AllocatorPing::template deallocate(vec_ping);
  AllocatorPong::template deallocate(vec_pong);

  return rc;
}

template <typename ValueType, typename ExecutionSpacePing,
          typename ExecutionSpacePong, typename ExecutionSpaceFirstTouchPing,
          typename ExecutionSpaceFirstTouchPong, typename AllocatorPingPong,
          typename IndexType>
auto benchmark_views(IndexType size, int warmups, int pingpongs,
                     AllocatorPingPong, NONE) {
  ValueType* vec_ping_pong =
      AllocatorPingPong::template allocate<ValueType>(size);

  auto rc = run_benchmark<Kokkos::SharedSpace, Kokkos::SharedSpace,
                          ExecutionSpacePing, ExecutionSpacePong,
                          ExecutionSpaceFirstTouchPing,
                          ExecutionSpaceFirstTouchPong, false>(
      vec_ping_pong, vec_ping_pong, size, warmups, pingpongs);

  AllocatorPingPong::template deallocate(vec_ping_pong);

  return rc;
}

template <typename ValueType, typename IndexType, typename AllocatorPing,
          typename AllocatorPong = NONE>
void benchmark_and_print(std::ostream& out, unsigned const rep,
                         IndexType array_size, unsigned warmups,
                         unsigned pingpongs, AllocatorPing Aping,
                         AllocatorPong Apong) {
  auto [rc, timing] = benchmark_views<ValueType, Kokkos::DefaultExecutionSpace,
                                      Kokkos::DefaultHostExecutionSpace,
                                      Kokkos::DefaultExecutionSpace,
                                      Kokkos::DefaultHostExecutionSpace>(
      array_size, warmups, pingpongs, Aping, Apong);
  if (rc != 0) {
    std::cout << "WRONG RESULT in rep " << rep << " array_size " << array_size
              << " warmups " << warmups << " pingpongs " << pingpongs
              << ".  exiting!" << std::endl;
    std::exit(rc);
  }

  double bw = 1.0e-6 * 2.0 * pingpongs * array_size *
              (double)sizeof(ValueType) / timing;
  out << rep << " , " << array_size << " , " << warmups << " , " << pingpongs
      << " , " << bw << " , " << typeid(AllocatorPing()).name() << " , "
      << typeid(AllocatorPong()).name() << "\n";
}

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  Kokkos::initialize(argc, argv);
  {
    using ValueType = int;
    using IndexType = int;

    if (argc < 8)
      printf(
          "Arguments: mode repetitions array_size array_size_steps "
          "warmup_runs warmpu_run_step ping_pongs ping_pong_step /n");

    const std::string mode(argv[1]);
    int repetitions           = std::stoi(argv[2]);
    IndexType array_size      = std::stoi(argv[3]);
    IndexType array_size_step = std::stoi(argv[4]);
    int warmup_runs           = std::stoi(argv[5]);
    int warmup_run_step       = std::stoi(argv[6]);
    int ping_pongs            = std::stoi(argv[7]);
    int ping_pong_step        = std::stoi(argv[8]);

    std::ofstream outfile;
    outfile.open(mode + ".csv", std::ios::out);

    Kokkos::print_configuration(outfile);

    outfile << "# repetition, arraysize, warmups, pingpongs, bandwidth, "
               "allocatorPing, "
               "allocatorPong"
            << std::endl;

    for (int pp = 1; pp <= ping_pongs; pp += ping_pong_step)
      for (int wu = 1; wu <= warmup_runs; wu += warmup_run_step)
        for (IndexType as = 1; as <= array_size; as += array_size_step)
          for (int rep = 0; rep <= repetitions; ++rep) {
            // TWO VIEWS
            // MANAGED
            if (mode == "managed-managed")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             ManagedMalloc(), ManagedMalloc());
            else if (mode == "managed-new")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             ManagedMalloc(), StdNew());
            else if (mode == "managed-malloc")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             ManagedMalloc(), StdMalloc());
            else if (mode == "managed-hostpinned")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             ManagedMalloc(),
                                             HostPinnedMalloc());
            else if (mode == "managed-device")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             ManagedMalloc(), DeviceMalloc());

            // DEVICE
            else if (mode == "device-managed")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), ManagedMalloc());
            else if (mode == "device-new")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), StdNew());
            else if (mode == "device-malloc")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), StdMalloc());
            else if (mode == "device-hostpinned")
              benchmark_and_print<ValueType>(
                  outfile, rep, as, wu, pp, DeviceMalloc(), HostPinnedMalloc());
            else if (mode == "device-device")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), DeviceMalloc());

            // HostPinned
            else if (mode == "hostpinned-managed")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), ManagedMalloc());
            else if (mode == "hostpinned-new")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), StdNew());
            else if (mode == "hostpinned-malloc")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), StdMalloc());
            else if (mode == "hostpinned-hostpinned")
              benchmark_and_print<ValueType>(
                  outfile, rep, as, wu, pp, DeviceMalloc(), HostPinnedMalloc());
            else if (mode == "hostpinned-device")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), DeviceMalloc());

            // NEW
            else if (mode == "new-managed")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp, StdNew(),
                                             ManagedMalloc());
            else if (mode == "new-new")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp, StdNew(),
                                             StdNew());
            else if (mode == "new-malloc")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp, StdNew(),
                                             StdMalloc());
            else if (mode == "new-hostpinned")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp, StdNew(),
                                             HostPinnedMalloc());
            else if (mode == "new-device")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp, StdNew(),
                                             DeviceMalloc());

            // ONE VIEW
            // MANAGED
            else if (mode == "managed-none")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             ManagedMalloc(), NONE());
            else if (mode == "hostpinned-none")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             ManagedMalloc(), NONE());
            else if (mode == "device-none")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             DeviceMalloc(), NONE());
            else if (mode == "malloc-none")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp,
                                             StdMalloc(), NONE());
            else if (mode == "new-none")
              benchmark_and_print<ValueType>(outfile, rep, as, wu, pp, StdNew(),
                                             NONE());
          }
    outfile.close();
  }
  Kokkos::finalize();

  return 0;
}
