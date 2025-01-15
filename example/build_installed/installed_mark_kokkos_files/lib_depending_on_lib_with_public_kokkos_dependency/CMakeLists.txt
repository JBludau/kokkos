add_library(
  ${PROJECT_NAME}_Lib_depending_on_lib_with_public_kokkos_dependency
  ../../example_source/lib_depending_on_lib_with_public_kokkos_dependency/fupp.cpp
)
# as the lib uses Kokkos in its interfaces, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_depending_on_lib_with_public_kokkos_dependency/fupp.cpp
)

target_link_libraries(
  ${PROJECT_NAME}_Lib_depending_on_lib_with_public_kokkos_dependency ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
)
# same for the target
kokkos_set_source_and_target_properties(TARGET ${PROJECT_NAME}_Lib_depending_on_lib_with_public_kokkos_dependency)
