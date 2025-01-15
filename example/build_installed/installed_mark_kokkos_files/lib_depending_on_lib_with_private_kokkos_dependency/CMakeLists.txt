add_library(
  ${PROJECT_NAME}_Lib_depending_on_lib_with_private_kokkos_dependency
  ../../example_source/lib_depending_on_lib_with_private_kokkos_dependency/full.cpp
)
target_link_libraries(${PROJECT_NAME}_Lib_depending_on_lib_with_private_kokkos_dependency PRIVATE ${PROJECT_NAME}_Lib_with_private_kokkos_dependency)
# target with private (tansitive) Kokkos dependency still have to be marked, as kokkos adds architecture dependent linker flags
kokkos_set_source_and_target_properties(LINK_ONLY_TARGET ${PROJECT_NAME}_Lib_depending_on_lib_with_private_kokkos_dependency)
