
cmake_minimum_required(VERSION 3.21)

project(nct)

add_subdirectory("../sfc" "${CMAKE_BINARY_DIR}/sfc")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tools)
