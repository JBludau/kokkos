set(Kokkos_GLOBAL_COMPILATION OFF)
find_package(Kokkos)

add_library(
  ${PROJECT_NAME}_Lib_with_private_kokkos_dependency
  ../../example_source/lib_with_private_kokkos_dependency/source_with_public_kokkos_dependency.cpp
  ../../example_source/lib_with_private_kokkos_dependency/source_without_kokkos_dependency.cpp
  ../../example_source/lib_with_private_kokkos_dependency/source_merging_prints.cpp
)

# manually mark files that use kokkos
kokkos_set_source_and_target_properties(
  SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_private_kokkos_dependency/source_with_public_kokkos_dependency.cpp
)

target_include_directories(
  ${PROJECT_NAME}_Lib_with_private_kokkos_dependency PUBLIC ../../example_source/lib_with_private_kokkos_dependency
)
target_link_libraries(${PROJECT_NAME}_Lib_with_private_kokkos_dependency PRIVATE Kokkos::kokkos)

# mark target as a kokkos target
kokkos_set_source_and_target_properties(TARGET ${PROJECT_NAME}_Lib_with_private_kokkos_dependency)

# mark all targets that link to the library downstream
kokkos_defer_set_dependent_library_properties(LIBRARY ${PROJECT_NAME}_Lib_with_private_kokkos_dependency)
