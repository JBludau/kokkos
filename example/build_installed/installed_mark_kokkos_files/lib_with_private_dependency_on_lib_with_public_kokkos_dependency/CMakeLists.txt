add_library(
  ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_public_kokkos_dependency
  ../../example_source/lib_with_private_dependency_on_lib_with_public_kokkos_dependency/source_with_private_dependency_on_lib_with_public_kokkos_dependency.cpp
)

# as the lib uses Kokkos in its interfaces, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
  SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_private_dependency_on_lib_with_public_kokkos_dependency/source_with_private_dependency_on_lib_with_public_kokkos_dependency.cpp
)

target_include_directories(
  ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_public_kokkos_dependency
  PUBLIC ../../example_source/lib_with_private_dependency_on_lib_with_public_kokkos_dependency
)
target_link_libraries(
  ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_public_kokkos_dependency
  PRIVATE ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
)

# same for the target
kokkos_set_source_and_target_properties(
  TARGET ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_public_kokkos_dependency
)
