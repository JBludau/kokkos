find_package(Kokkos)
# disable kokkos setting source and target properties automatically. We want to do this manually.
# this implies that other libs in this poject can not rely on kokkos setting the properties automatically
# see `lib_with_public_kokkos_dependency` for an example how to do this
kokkos_exclude_from_setting_build_properties(GLOBAL)

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

# build all targets with kokkos settings if they link to the current library downstream
kokkos_defer_set_dependent_library_properties(LIBRARY ${PROJECT_NAME}_Lib_with_private_kokkos_dependency)
