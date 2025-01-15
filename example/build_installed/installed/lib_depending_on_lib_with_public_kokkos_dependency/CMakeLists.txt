add_library(
  ${PROJECT_NAME}_Lib_depending_on_lib_with_public_kokkos_dependency
  ../../example_source/lib_depending_on_lib_with_public_kokkos_dependency/fupp.cpp
)

target_link_libraries(
  ${PROJECT_NAME}_Lib_depending_on_lib_with_public_kokkos_dependency ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
)
