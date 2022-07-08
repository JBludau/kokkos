
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

#include <iostream>

#include <Kokkos_Core.hpp>

using DeviceExecSpace = Kokkos::Experimental::HIP;
using HostExecSpace   = Kokkos::Serial;

#if defined KOKKOS_ENABLE_HIP
using PageMigratingMemorySpace = Kokkos::Experimental::HIPManagedSpace;
#elif defined KOKKOS_ENABLE_CUDA
using PageMigratingMemorySpace = Kokkos::CudaUVMSpace;
#elif defined KOKKOS_ENABLE_SYCL
using PageMigratingMemorySpace = Kokkos::Experimental::SyclSharedUSMSpace;
#endif

void printTimings(std::ostream& out, std::vector<double> tr) {
  out << "TimingResult contains " << tr.size() << " results:\n";
  for (auto it = tr.begin(); it != tr.end(); ++it) {
    out << "Duration of loop " << it - tr.begin() << " is " << *it
        << " seconds\n";
  }
}

template <typename ExecSpace, typename ViewType>
std::vector<decltype(std::declval<Kokkos::Timer>().seconds())> incrementInLoop(
    ViewType& view, unsigned int loopCount) {
  Kokkos::Timer timer;
  std::vector<decltype(timer.seconds())> results;

  for (unsigned i = 0; i < loopCount; ++i) {
    auto start = timer.seconds();
    Kokkos::parallel_for(
        "increment", Kokkos::RangePolicy<ExecSpace>(0, view.size()),
        KOKKOS_LAMBDA(const int& i) { ++view(i); });
    Kokkos::fence();
    auto end = timer.seconds();
    results.push_back(end - start);
  }
  return results;
}

size_t getDeviceMemory() {
#if defined KOKKOS_ENABLE_CUDA
  return Kokkos::Cuda::cuda_device_prop().totalGlobalMem;
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
    double memoryFraction = 0.5;
    size_t noBytes        = memoryFraction * getDeviceMemory();
    unsigned int noPages  = noBytes / getBytesPerPage();
    std::cout << "Page size as reported by os: " << getBytesPerPage()
              << " bytes \n";
    std::cout << "Allocating " << noPages
              << " pages of memory in pageMigratingMemorySpace.\n"
              << "This corresponds to " << memoryFraction * 100
              << " % of the device memory.\n";

    Kokkos::View<int*, PageMigratingMemorySpace> migratableData(
        "migratableData", noPages * getBytesPerPage() / sizeof(int));
    Kokkos::View<int*, DeviceExecSpace::memory_space> deviceData(
        "migratableData", 0.5 * noPages * getBytesPerPage() / sizeof(int));

    incrementInLoop<DeviceExecSpace>(deviceData, 10);  // warming up gpu
    for (unsigned i = 0; i < 3; ++i) {
      auto deviceResults = incrementInLoop<DeviceExecSpace>(migratableData, 10);
      std::cout << "device run " << i << ":\n";
      printTimings(std::cout, deviceResults);
      auto hostResults = incrementInLoop<HostExecSpace>(migratableData, 10);
      std::cout << "host run " << i << ":\n";
      printTimings(std::cout, hostResults);
    }
  }
  Kokkos::finalize();
  return 0;
}
