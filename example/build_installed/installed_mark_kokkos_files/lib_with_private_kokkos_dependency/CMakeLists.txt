find_package(Kokkos)

add_library(
  ${PROJECT_NAME}_Lib_with_private_kokkos_dependency ../../example_source/lib_with_private_kokkos_dependency/source_with_public_kokkos_dependency.cpp ../../example_source/lib_with_private_kokkos_dependency/source_without_kokkos_dependency.cpp
 ../../example_source/lib_with_private_kokkos_dependency/source_merging_prints.cpp)
target_include_directories(${PROJECT_NAME}_Lib_with_private_kokkos_dependency PUBLIC ../../example_source/lib_with_private_kokkos_dependency)
target_link_libraries(${PROJECT_NAME}_Lib_with_private_kokkos_dependency PRIVATE Kokkos::kokkos)
