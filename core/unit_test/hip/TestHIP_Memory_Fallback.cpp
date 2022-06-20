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

namespace Test {

template <typename T>
__global__ void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    C_d[i] = A_d[i] + B_d[i];
  }
}

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

int N = 64 * 1024 * 1024;
TEST(hip, memory_fallback) {
  printDeviceInfo(0);
  printMemoryInformation<double>(hipMallocManaged, 10);
  printMemoryInformation<double>(hipHostMalloc, 10);
  int numElements = (N < (64 * 1024 * 1024)) ? 64 * 1024 * 1024 : N;
  float *A, *B, *C;

  KOKKOS_IMPL_HIP_SAFE_CALL(hipMallocManaged(&A, numElements * sizeof(float)));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMallocManaged(&B, numElements * sizeof(float)));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMallocManaged(&C, numElements * sizeof(float)));

  hipDevice_t device = hipCpuDeviceId;

  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemAdvise(A, numElements * sizeof(float),
                                         hipMemAdviseSetReadMostly, device));
  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipMemPrefetchAsync(A, numElements * sizeof(float), 0));
  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipMemPrefetchAsync(B, numElements * sizeof(float), 0));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceSynchronize());
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemRangeGetAttribute(
      &device, sizeof(device), hipMemRangeAttributeLastPrefetchLocation, A,
      numElements * sizeof(float)));
  if (device != 0) {
    std::cout << "hipMemRangeGetAttribute error, device = " << device << "\n";
  }
  uint32_t read_only = 0xf;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemRangeGetAttribute(
      &read_only, sizeof(read_only), hipMemRangeAttributeReadMostly, A,
      numElements * sizeof(float)));
  if (read_only != 1) {
    std::cout << "hipMemRangeGetAttribute error, read_only = " << read_only
              << "\n";
  }

  unsigned blocks = numElements;

  hipEvent_t event0, event1;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipEventCreate(&event0));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipEventCreate(&event1));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipEventRecord(event0, 0));
  hipLaunchKernelGGL(vectorADD, dim3(blocks), dim3(1), 0, 0,
                     static_cast<const float*>(A), static_cast<const float*>(B),
                     C, numElements);
  KOKKOS_IMPL_HIP_SAFE_CALL(hipEventRecord(event1, 0));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceSynchronize());
  float time = 0.0f;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipEventElapsedTime(&time, event0, event1));
  printf("Time %.3f ms\n", time);
  float maxError = 0.0f;
  KOKKOS_IMPL_HIP_SAFE_CALL(
      hipMemPrefetchAsync(B, numElements * sizeof(float), hipCpuDeviceId));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceSynchronize());
  device = 0;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemRangeGetAttribute(
      &device, sizeof(device), hipMemRangeAttributeLastPrefetchLocation, A,
      numElements * sizeof(float)));
  if (device != hipCpuDeviceId) {
    std::cout << "hipMemRangeGetAttribute error device = " << device << "\n";
  }

  for (int i = 0; i < numElements; i++) {
    maxError = fmax(maxError, fabs(B[i] - 3.0f));
  }
  KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(A));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(B));
  std::cout << "maxError: " << maxError << std::endl;
  ;
}

}  // namespace Test
