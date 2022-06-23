
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

#include <TestHIP_Category.hpp>

namespace Test {

template <typename T>
hipMemRangeCoherencyMode checkMemoryGrainedness(
    hipError_t alloc(void**, size_t, unsigned int),
    bool memAdviseCoarseGrain = false) {
  T* ptr = nullptr;
  hipMemRangeCoherencyMode memInfo;
  KOKKOS_IMPL_HIP_SAFE_CALL(
      alloc((void**)(&ptr), 1 * sizeof(T), hipMemAttachGlobal));
  if (memAdviseCoarseGrain) {
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipMemAdvise(ptr, 1 * sizeof(T), hipMemAdviseSetCoarseGrain, 0));
  }

  KOKKOS_IMPL_HIP_SAFE_CALL(hipMemRangeGetAttribute(
      &memInfo, 1 * sizeof(hipMemRangeCoherencyMode),
      hipMemRangeAttributeCoherencyMode, ptr, 1 * sizeof(T)));
  KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(ptr));
  return memInfo;
}

TEST(hip, memory_requirements) {
  ASSERT_TRUE(hipMemRangeCoherencyModeCoarseGrain ==
              checkMemoryGrainedness<int>(hipHostMalloc, false));
  ASSERT_FALSE(hipMemRangeCoherencyModeCoarseGrain ==
               checkMemoryGrainedness<int>(hipMallocManaged, false));
  ASSERT_TRUE(hipMemRangeCoherencyModeCoarseGrain ==
              checkMemoryGrainedness<int>(hipMallocManaged, true));
}
}  // end of namespace Test
