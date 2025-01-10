find_package(Kokkos)

add_library(Lib_with_private_kokkos_dependency ../../example_source/lib_with_private_kokkos_dependency/furr.cpp)
target_link_libraries(Lib_with_private_kokkos_dependency PRIVATE Kokkos::kokkos)
