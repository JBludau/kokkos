set(Kokkos_GLOBAL_COMPILATION OFF)
find_package(Kokkos)

add_library(Lib_with_public_kokkos_dependency ../../example_source/lib_with_public_kokkos_dependency/fuzz.cpp)
# manually mark files that use kokkos
kokkos_set_source_and_target_properties(
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_public_kokkos_dependency/fuzz.cpp
)

target_link_libraries(Lib_with_public_kokkos_dependency PUBLIC Kokkos::kokkos)
# mark target as a kokkos target
kokkos_set_source_and_target_properties(TARGET Lib_with_public_kokkos_dependency)

# mark all targets that link to the library downstream
kokkos_defer_set_dependent_library_properties(PUBLIC LIBRARY Lib_with_public_kokkos_dependency)
