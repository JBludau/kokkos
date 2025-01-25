find_package(Kokkos)

add_library(
  ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
  ../../example_source/lib_with_public_kokkos_dependency/source_with_public_kokkos_dependency.cpp
)

target_include_directories(
  ${PROJECT_NAME}_Lib_with_public_kokkos_dependency PUBLIC ../../example_source/lib_with_public_kokkos_dependency
)
target_link_libraries(${PROJECT_NAME}_Lib_with_public_kokkos_dependency PUBLIC Kokkos::kokkos)

# mark dir as having kokkos targets
kokkos_set_source_and_target_properties(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# build all targets with kokkos settings if they link to the library downstream
kokkos_defer_set_dependent_library_properties(PUBLIC LIBRARY ${PROJECT_NAME}_Lib_with_public_kokkos_dependency)
