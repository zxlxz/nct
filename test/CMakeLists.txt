
file(GLOB_RECURSE SRC "${CMAKE_CURRENT_LIST_DIR}/**.cxx")

add_executable(nct_test ${SRC})
target_compile_features(nct_test PUBLIC cxx_std_20)
target_link_libraries(nct_test PRIVATE nct)
