find_package(Kokkos)

add_library(${PROJECT_NAME}_Lib_with_public_kokkos_dependency ../../example_source/lib_with_public_kokkos_dependency/fuzz.cpp)
target_link_libraries(${PROJECT_NAME}_Lib_with_public_kokkos_dependency PUBLIC Kokkos::kokkos)
