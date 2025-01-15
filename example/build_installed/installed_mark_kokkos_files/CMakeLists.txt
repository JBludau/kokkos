# Kokkos minimally requires 3.19 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.19)

# Projects can safely mix languages - must have C++ support
project(ExampleInstalledMarkKokkosFiles CXX Fortran)

# In the top-level CMakeLists.txt ensure, that all find_package(Kokkos) calls are done without setting compilation flags for the whole project
set(Kokkos_GLOBAL_COMPILATION OFF)
find_package(Kokkos)

# define some libs in subdirs
add_subdirectory(lib_without_kokkos_dependency)

# targets that depend on kokkos have to be marked manually. This can be done at e.g. directory level
add_subdirectory(lib_with_public_kokkos_dependency)
kokkos_set_source_and_target_properties(DIRECTORY lib_with_public_kokkos_dependency)
add_subdirectory(lib_with_private_kokkos_dependency)
kokkos_set_source_and_target_properties(DIRECTORY lib_with_private_kokkos_dependency)

# or more fine grained at source and target level in the respective CMakeLists.txt of the libs
add_subdirectory(lib_depending_on_lib_with_public_kokkos_dependency)
add_subdirectory(lib_depending_on_lib_with_private_kokkos_dependency)

# create an executable that uses kokkos in interfaces
add_executable(
  ${PROJECT_NAME}_executable_public_kokkos ../example_source/example_public_kokkos.cpp ../example_source/bar.cpp ../example_source/foo.f
)
# as kokkos is used in the interface, the files and target have to be marked
kokkos_set_source_and_target_properties(
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/example_public_kokkos.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/bar.cpp
)
kokkos_set_source_and_target_properties(TARGET ${PROJECT_NAME}_executable_public_kokkos)

# create an executable that does not use kokkos in interfaces
add_executable(
  ${PROJECT_NAME}_executable_private_kokkos ../example_source/example_private_kokkos.cpp ../example_source/bar.cpp ../example_source/foo.f
)
# as kokkos adds architecture dependent linker flags, targets with (transitive) private dependency on Kokkos have to be marked as well
kokkos_set_source_and_target_properties(LINK_ONLY_TARGET ${PROJECT_NAME}_executable_private_kokkos)

# for LLVMFlang we need to use fortran for linking
if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public_kokkos PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(${PROJECT_NAME}_executable_private_kokkos PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_options(${PROJECT_NAME}_executable_public_kokkos PRIVATE -fno-fortran-main)
  target_link_options(${PROJECT_NAME}_executable_private_kokkos PRIVATE -fno-fortran-main)
endif()

#link your executable to the libraries
target_link_libraries(
  ${PROJECT_NAME}_executable_public_kokkos ${PROJECT_NAME}_Lib_without_kokkos_dependency ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_private_kokkos_dependency ${PROJECT_NAME}_Lib_depending_on_lib_with_public_kokkos_dependency
  ${PROJECT_NAME}_Lib_depending_on_lib_with_public_kokkos_dependency
)

target_link_libraries(
  ${PROJECT_NAME}_executable_private_kokkos ${PROJECT_NAME}_Lib_without_kokkos_dependency ${PROJECT_NAME}_Lib_with_private_kokkos_dependency
  ${PROJECT_NAME}_Lib_depending_on_lib_with_private_kokkos_dependency
)

enable_testing()
add_test(NAME ${PROJECT_NAME}_KokkosPublic_Verify COMMAND ${PROJECT_NAME}_executable_public_kokkos 10)
add_test(NAME ${PROJECT_NAME}_KokkosPrivate_Verify COMMAND ${PROJECT_NAME}_executable_private_kokkos 10)
