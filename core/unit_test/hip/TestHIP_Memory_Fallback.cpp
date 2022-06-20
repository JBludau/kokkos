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

#include <Kokkos_Core.hpp>
#include <TestHIP_Category.hpp>
#include "Kokkos_View.hpp"

namespace Test {

std::ostream& operator<<(std::ostream& os, hipPointerAttribute_t const attr) {
  os << "hipPointerAttribute: \n";
  os << "hipMemoryType: " << attr.memoryType << "\n";
  os << "device: " << attr.device << "\n";
  os << "isManaged: " << attr.isManaged << "\n";
  os << "allocationFlags: " << attr.allocationFlags << "\n";
  return os;
}

void printDeviceInfo(int deviceNo) {
  int managed;
  std::cout << "The following are the attribute values related to HMM for"
               " device "
            << deviceNo << ":\n";
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceGetAttribute(
      &managed, hipDeviceAttributeDirectManagedMemAccessFromHost, deviceNo));
  std::cout << "hipDeviceAttributeDirectManagedMemAccessFromHost: " << managed
            << "\n";
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceGetAttribute(
      &managed, hipDeviceAttributeConcurrentManagedAccess, deviceNo));
  std::cout << "hipDeviceAttributeConcurrentManagedAccess: " << managed << "\n";
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceGetAttribute(
      &managed, hipDeviceAttributePageableMemoryAccess, deviceNo));
  std::cout << "hipDeviceAttributePageableMemoryAccess: " << managed << "\n";
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceGetAttribute(
      &managed, hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
      deviceNo));
  std::cout << "hipDeviceAttributePageableMemoryAccessUsesHostPageTables:"
            << managed << "\n";

  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceGetAttribute(
      &managed, hipDeviceAttributeManagedMemory, deviceNo));
  std::cout << "hipDeviceAttributeManagedMemory: " << managed << "\n"
            << std::endl;
}

template <typename T>
void printMemoryInformation(hipError_t alloc(void**, size_t, unsigned int),
                            unsigned int size) {
  T* ptr = nullptr;
  KOKKOS_IMPL_HIP_SAFE_CALL(alloc((void**)(&ptr), size, hipMemAttachGlobal));
  hipPointerAttribute_t attr;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipPointerGetAttributes(&attr, ptr));
  std::cout << attr << std::endl;
  unsigned int flags;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipHostGetFlags(&flags, ptr));
  std::cout << "hipHostGetFlags: \n";
  std::cout << "flags: " << flags << "\n\n";
  KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(ptr));
}

TEST(hip, memory_fallback) {
  printDeviceInfo(0);
  printMemoryInformation<double>(hipMallocManaged, 10);
  printMemoryInformation<double>(hipHostMalloc, 10);
}

}  // namespace Test
