
file(GLOB_RECURSE SRC "${CMAKE_CURRENT_LIST_DIR}/src/**.cxx")

add_executable(nct_test ${SRC})
target_link_libraries(nct_test sfc nct)
