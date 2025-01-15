# Kokkos minimally requires 3.19 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.19)

# Projects can safely mix languages - must have C++ support
project(BuildExamples CXX)

add_subdirectory(installed)
add_subdirectory(installed_mark_kokkos_files)
add_subdirectory(installed_mark_library)
