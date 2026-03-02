kokkos_cfg_depends(TPLS OPTIONS)
kokkos_cfg_depends(TPLS DEVICES)
kokkos_cfg_depends(TPLS COMPILER_ID)

function(KOKKOS_TPL_OPTION PKG DEFAULT)
  cmake_parse_arguments(PARSED "" "TRIBITS" "" ${ARGN})

  if(PARSED_TRIBITS)
    #this is also a TPL option you can activate with Tribits
    if(NOT "${TPL_ENABLE_${PARSED_TRIBITS}}" STREQUAL "")
      #Tribits brought its own default that should take precedence
      set(DEFAULT ${TPL_ENABLE_${PARSED_TRIBITS}})
    endif()
  endif()

  kokkos_enable_option(${PKG} ${DEFAULT} "Whether to enable the ${PKG} library")
  kokkos_option(${PKG}_DIR "" PATH "Location of ${PKG} library")
  set(KOKKOS_ENABLE_${PKG} ${KOKKOS_ENABLE_${PKG}} PARENT_SCOPE)
  set(KOKKOS_${PKG}_DIR ${KOKKOS_${PKG}_DIR} PARENT_SCOPE)
endfunction()

kokkos_tpl_option(HWLOC Off TRIBITS HWLOC)
kokkos_tpl_option(CUDA ${Kokkos_ENABLE_CUDA} TRIBITS CUDA)
kokkos_tpl_option(ROCM ${Kokkos_ENABLE_HIP})
kokkos_tpl_option(ROCTHRUST ${Kokkos_ENABLE_HIP})

# FIXME_HIP
# This is here to move our first find_package(hip) call to after we processed the arch detection.
# The origin of the problem is that find_package(hip) does an automatic detection of GPU archs into GPU_TARGETS.
# Since MALLOC_ASYNC is enabled based on the hip version, it is moved here to not require an earlier find_package(hip) call
if(KOKKOS_ENABLE_HIP)
  if(hip_VERSION VERSION_GREATER_EQUAL 7.0.0)
    set(HIP_MALLOC_ASYNC_DEFAULT OFF)
  else()
    set(HIP_MALLOC_ASYNC_DEFAULT ${KOKKOS_ENABLE_HIP})
  endif()
  kokkos_enable_option(IMPL_HIP_MALLOC_ASYNC ${HIP_MALLOC_ASYNC_DEFAULT} "Whether to enable hipMallocAsync")
  if((hip_VERSION VERSION_GREATER_EQUAL 7.0.0) AND Kokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC)
    message(WARNING "Using Kokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC is problematic with ROCm 7")
  endif()
endif()

if(KOKKOS_ENABLE_SYCL)
  set(ONEDPL_DEFAULT ON)
else()
  set(ONEDPL_DEFAULT OFF)
endif()
kokkos_tpl_option(ONEDPL ${ONEDPL_DEFAULT})

if(WIN32)
  set(LIBDL_DEFAULT Off)
else()
  set(LIBDL_DEFAULT On)
endif()
kokkos_enable_option(LIBDL ${LIBDL_DEFAULT} "Whether to enable the LIBDL library")

if(Trilinos_ENABLE_Kokkos AND TPL_ENABLE_HPX)
  set(HPX_DEFAULT ON)
else()
  set(HPX_DEFAULT OFF)
endif()
kokkos_tpl_option(HPX ${HPX_DEFAULT})

kokkos_tpl_option(THREADS ${Kokkos_ENABLE_THREADS} TRIBITS Pthread)

if(Trilinos_ENABLE_Kokkos AND TPL_ENABLE_quadmath)
  set(LIBQUADMATH_DEFAULT ON)
else()
  set(LIBQUADMATH_DEFAULT OFF)
endif()
kokkos_tpl_option(LIBQUADMATH ${LIBQUADMATH_DEFAULT} TRIBITS quadmath)

#Make sure we use our local FindKokkosCuda.cmake
kokkos_import_tpl(HPX INTERFACE)
kokkos_import_tpl(CUDA INTERFACE)
kokkos_import_tpl(HWLOC)
if(NOT WIN32)
  kokkos_import_tpl(THREADS INTERFACE)
endif()
if(NOT KOKKOS_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
  kokkos_import_tpl(ROCM INTERFACE)
endif()
kokkos_import_tpl(ONEDPL INTERFACE)
kokkos_import_tpl(LIBQUADMATH)
kokkos_import_tpl(ROCTHRUST)

if(Kokkos_ENABLE_DESUL_ATOMICS_EXTERNAL)
  find_package(desul REQUIRED COMPONENTS atomics)
  kokkos_export_cmake_tpl(desul REQUIRED COMPONENTS atomics)
endif()

if(Kokkos_ENABLE_IMPL_MDSPAN AND Kokkos_ENABLE_MDSPAN_EXTERNAL)
  find_package(mdspan REQUIRED)
  kokkos_export_cmake_tpl(mdspan REQUIRED)
endif()

if(Kokkos_ENABLE_OPENMP)
  find_package(OpenMP 3.0 REQUIRED COMPONENTS CXX)
  kokkos_export_cmake_tpl(OpenMP REQUIRED COMPONENTS CXX)
  if(Kokkos_ENABLE_HIP AND KOKKOS_COMPILE_LANGUAGE STREQUAL HIP)
    global_append(KOKKOS_AMDGPU_OPTIONS ${OpenMP_CXX_FLAGS})
  endif()
  if(Kokkos_ENABLE_CUDA AND KOKKOS_COMPILE_LANGUAGE STREQUAL CUDA)
    if(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
      global_append(KOKKOS_CUDA_OPTIONS -Xcompiler ${OpenMP_CXX_FLAGS})
    else()
      global_append(KOKKOS_CUDA_OPTIONS ${OpenMP_CXX_FLAGS})
    endif()
  endif()
endif()

#Convert list to newlines (which CMake doesn't always like in cache variables)
string(REPLACE ";" "\n" KOKKOS_TPL_EXPORT_TEMP "${KOKKOS_TPL_EXPORTS}")
#Convert to a regular variable
unset(KOKKOS_TPL_EXPORTS CACHE)
set(KOKKOS_TPL_EXPORTS ${KOKKOS_TPL_EXPORT_TEMP})
