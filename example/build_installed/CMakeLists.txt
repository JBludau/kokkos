# Kokkos minimally requires 3.19 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.19)

# Projects can safely mix languages - must have C++ support
project(BuildExamples CXX)

# IMPORTANT INFORMATION
# this file is only for dev builds
# it does not reflect how to use the build features of kokkos

if(Examples_CMAKE_LANGUAGE)
  enable_language(${Examples_CMAKE_LANGUAGE})
endif()

set(Kokkos_EXCLUDE_BUILD_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/installed_mark_kokkos_files;${CMAKE_CURRENT_SOURCE_DIR}/installed_mark_library"
    CACHE STRING "List of directories Kokkos that are excluded from Kokkos setting build properties on source and target properties via a deferred call"
  )

add_subdirectory(installed)
add_subdirectory(installed_mark_kokkos_files)
add_subdirectory(installed_mark_library)
