find_package(Kokkos)
# exclude current dir from kokkos setting source and target properties automatically. We want to do this manually.
kokkos_exclude_from_setting_build_properties(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(
  ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
  ../../example_source/lib_with_public_kokkos_dependency/source_with_public_kokkos_dependency.cpp
)

# manually mark files that use kokkos
kokkos_set_source_and_target_properties(
  SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_public_kokkos_dependency/source_with_public_kokkos_dependency.cpp
)

target_include_directories(
  ${PROJECT_NAME}_Lib_with_public_kokkos_dependency PUBLIC ../../example_source/lib_with_public_kokkos_dependency
)
target_link_libraries(${PROJECT_NAME}_Lib_with_public_kokkos_dependency PUBLIC Kokkos::kokkos)

# mark target as a kokkos target
kokkos_set_source_and_target_properties(TARGET ${PROJECT_NAME}_Lib_with_public_kokkos_dependency)

# build all targets with kokkos settings if they link to the library downstream
kokkos_defer_set_dependent_library_properties(PUBLIC LIBRARY ${PROJECT_NAME}_Lib_with_public_kokkos_dependency)
