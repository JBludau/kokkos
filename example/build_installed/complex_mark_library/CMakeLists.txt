# Kokkos minimally requires 3.19 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.19)

# Projects can safely mix languages - must have C++ support
project(ExampleComplexMarkLibrary CXX Fortran)

# define some libs in subdirs
add_subdirectory(lib_without_kokkos_dependency)
add_subdirectory(lib_with_public_kokkos_dependency)
add_subdirectory(lib_with_private_kokkos_dependency)
add_subdirectory(lib_with_private_dependency_on_lib_with_private_kokkos_dependency)
add_subdirectory(lib_with_private_dependency_on_lib_with_public_kokkos_dependency)
add_subdirectory(lib_with_public_dependency_on_lib_with_private_kokkos_dependency)
add_subdirectory(lib_with_public_dependency_on_lib_with_public_kokkos_dependency)

# create an executable that uses kokkos in interfaces
add_executable(
  ${PROJECT_NAME}_executable_public_kokkos
  ../example_source/example_public_kokkos.cpp ../example_source/source_with_kokkos_dependency.cpp
  ../example_source/source_plain_fortran.f
)

# create an executable that does not use kokkos in interfaces
add_executable(
  ${PROJECT_NAME}_executable_private_kokkos
  ../example_source/example_private_kokkos.cpp ../example_source/source_plain_cxx.cpp
  ../example_source/source_plain_fortran.f
)

# for LLVMFlang we need to use fortran for linking
if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public_kokkos PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(${PROJECT_NAME}_excutable_private_kokkos PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_options(${PROJECT_NAME}_executable_public_kokkos PRIVATE -fno-fortran-main)
  target_link_options(${PROJECT_NAME}_executale_private_kokkos PRIVATE -fno-fortran-main)
endif()

#link your executable to the libraries
target_link_libraries(
  ${PROJECT_NAME}_executable_private_kokkos
  ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_private_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_private_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_public_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_private_kokkos_dependency
)
target_link_libraries(
  ${PROJECT_NAME}_executable_public_kokkos ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_public_kokkos_dependency
)

enable_testing()
add_test(NAME ${PROJECT_NAME}_KokkosPublic_Verify COMMAND ${PROJECT_NAME}_executable_public_kokkos 10)
add_test(NAME ${PROJECT_NAME}_KokkosPrivate_Verify COMMAND ${PROJECT_NAME}_executable_private_kokkos 10)
