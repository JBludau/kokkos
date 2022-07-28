
/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#if defined _WIN32  // windows system
#include <windows.h>
unsigned getBytesPerPage() {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
}
#else  // unix/posix system
#include <unistd.h>
unsigned getBytesPerPage() { return sysconf(_SC_PAGESIZE); }
#endif

#include <algorithm>
#include <iostream>

#include <Kokkos_Core.hpp>

#if defined KOKKOS_ENABLE_SERIAL
using HostExecSpace = Kokkos::Serial;
#elif defined KOKKOS_ENABLE_OPENMP
using HostExecSpace            = Kokkos::OpenMP;
#endif

#if defined KOKKOS_ENABLE_CUDA
using PageMigratingMemorySpace = Kokkos::CudaUVMSpace;
using DeviceExecSpace          = Kokkos::Cuda;
#elif defined KOKKOS_ENABLE_HIP
using PageMigratingMemorySpace = Kokkos::Experimental::HIPManagedSpace;
using DeviceExecSpace          = Kokkos::Experimental::HIP;
#elif defined KOKKOS_ENABLE_SYCL
using PageMigratingMemorySpace = Kokkos::Experimental::SyclSharedUSMSpace;
using DeviceExecSpace          = Kokkos::Experimental::Sycl;
#endif

void printTimings(std::ostream& out, std::vector<double> tr) {
  out << "TimingResult contains " << tr.size() << " results:\n";
  for (auto it = tr.begin(); it != tr.end(); ++it) {
    out << "Duration of loop " << it - tr.begin() << " is " << *it
        << " seconds\n";
  }
}

template <typename T>
T computeMean(std::vector<T> results) {
  T res{};
  for (auto it = results.begin(); it != results.end(); ++it) {
    res += *it;
  }
  return res / results.size();
}

template <typename ExecSpace, typename ViewType>
std::vector<decltype(std::declval<Kokkos::Timer>().seconds())> incrementInLoop(
    ViewType& view, unsigned int loopCount) {
  Kokkos::Timer timer;
  std::vector<decltype(timer.seconds())> results;

  for (unsigned i = 0; i < loopCount; ++i) {
    auto start = timer.seconds();
    Kokkos::parallel_for(
        "increment",
        Kokkos::RangePolicy<ExecSpace>{0,
                                       static_cast<unsigned int>(view.size())},
        KOKKOS_LAMBDA(const int64_t& i) { ++view(i); });
    Kokkos::fence();
    auto end = timer.seconds();
    results.push_back(end - start);
  }
  return results;
}

size_t getDeviceMemorySize() {
#if defined KOKKOS_ENABLE_CUDA
  return Kokkos::Cuda{}.cuda_device_prop().totalGlobalMem;
#elif defined KOKKOS_ENABLE_HIP
  return Kokkos::Experimental::HIP::hip_device_prop().totalGlobalMem;
#else
  static_assert(false, "Only implemented for HIP and Cuda");
  return 0;
#endif
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const unsigned int noRepetitions      = 10;
    const unsigned int noDeviceHostCycles = 3;
    double fractionOfDeviceMemory         = 0.4;
    double threshold                      = 2.0;
    size_t noBytes       = fractionOfDeviceMemory * getDeviceMemorySize();
    unsigned int noPages = noBytes / getBytesPerPage();

    // ALLOCATION
    Kokkos::View<int*, PageMigratingMemorySpace> migratableData(
        "migratableData", noPages * getBytesPerPage() / sizeof(int));
    Kokkos::View<int*, DeviceExecSpace::memory_space> deviceData(
        "deviceData", noPages * getBytesPerPage() / sizeof(int));
    Kokkos::View<int*, HostExecSpace::memory_space> hostData(
        "hostData", noPages * getBytesPerPage() / sizeof(int));
    Kokkos::fence();

    // WARMUP GPU
    incrementInLoop<DeviceExecSpace>(deviceData,
                                     noRepetitions);  // warming up gpu

    // GET DEVICE LOCAL TIMINGS
    auto deviceLocalResults =
        incrementInLoop<DeviceExecSpace>(deviceData, 10 * noRepetitions);

    // WARMUP HOST
    incrementInLoop<HostExecSpace>(hostData,
                                   10 * noRepetitions);  // warming up host
    // GET HOST LOCAL TIMINGS
    auto hostLocalResults =
        incrementInLoop<HostExecSpace>(hostData, noRepetitions);

    // GET PAGE MIGRATING TIMINGS DATA
    std::vector<decltype(deviceLocalResults)> deviceResults{};
    std::vector<decltype(hostLocalResults)> hostResults{};
    for (unsigned i = 0; i < noDeviceHostCycles; ++i) {
      // WARMUP GPU
      incrementInLoop<DeviceExecSpace>(deviceData,
                                       10 * noRepetitions);  // warming up gpu
      // GET RESULTS DEVICE
      deviceResults.push_back(
          incrementInLoop<DeviceExecSpace>(migratableData, noRepetitions));

      // WARMUP HOST
      incrementInLoop<HostExecSpace>(hostData,
                                     10 * noRepetitions);  // warming up host
      // GET RESULTS HOST
      hostResults.push_back(
          incrementInLoop<HostExecSpace>(migratableData, noRepetitions));
    }

    // COMPUTE STATISTICS OF HOST AND DEVICE LOCAL KERNELS
    auto hostLocalMean   = computeMean(hostLocalResults);
    auto deviceLocalMean = computeMean(deviceLocalResults);

    // ASSESS PAGE MIGRATIONS
    bool initialPlacementOnDevice   = false;
    bool migratesOnEverySpaceAccess = true;
    bool migratesOnlyOncePerAccess  = true;

    for (unsigned cycle = 0; cycle < noDeviceHostCycles; ++cycle) {
      unsigned int indicatedPageMigrationsDevice = std::count_if(
          deviceResults[cycle].begin(), deviceResults[cycle].end(),
          [&](auto const& val) { return val > (threshold * deviceLocalMean); });

      if (cycle == 0 && indicatedPageMigrationsDevice == 0)
        initialPlacementOnDevice = true;
      else {
        if (indicatedPageMigrationsDevice != 1)
          migratesOnlyOncePerAccess = false;
      }

      unsigned int indicatedPageMigrationsHost = std::count_if(
          hostResults[cycle].begin(), hostResults[cycle].end(),
          [&](auto const& val) { return val > (threshold * hostLocalMean); });

      if (indicatedPageMigrationsHost != 1) migratesOnlyOncePerAccess = false;

      if (cycle != 0 && indicatedPageMigrationsDevice != 1 &&
          indicatedPageMigrationsHost != 1)
        migratesOnEverySpaceAccess = false;
    }

    // CHECK IF PASSED
    bool passed = (initialPlacementOnDevice && migratesOnEverySpaceAccess &&
                   migratesOnlyOncePerAccess);

    // PRINT IF NOT PASSED
    if (!passed) {
      std::cout << "Page size as reported by os: " << getBytesPerPage()
                << " bytes \n";
      std::cout << "Allocating " << noPages
                << " pages of memory in pageMigratingMemorySpace.\n"
                << "This corresponds to " << fractionOfDeviceMemory * 100
                << " % of the device memory.\n\n";

      std::cout << "Behavior found: \n";
      std::cout << "Initial placement on device is " << initialPlacementOnDevice
                << " we expect true \n";
      std::cout << "Memory migrates on every space access is "
                << migratesOnEverySpaceAccess << " we expect true \n";
      std::cout << "Memory migrates only once per access "
                << migratesOnlyOncePerAccess << " we expect true \n\n";

      std::cout << "Please look at the following timings. A migration was "
                   "marked detected if the time was larger than "
                << threshold * hostLocalMean << " for the host and "
                << threshold * deviceLocalMean << " for the device \n\n";

      for (unsigned cycle = 0; cycle < noDeviceHostCycles; ++cycle) {
        std::cout << "device timings of run " << cycle << ":\n";
        printTimings(std::cout, deviceResults[cycle]);
        std::cout << "host timings of run " << cycle << ":\n";
        printTimings(std::cout, hostResults[cycle]);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
