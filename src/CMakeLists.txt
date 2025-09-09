find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

file(GLOB_RECURSE CC_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/**.cc")
file(GLOB_RECURSE CU_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/**.cu")
add_library(nct ${CC_SRC} ${CU_SRC})

target_compile_features(nct PUBLIC cxx_std_23)
target_include_directories(nct PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(nct PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(nct sfc CUDA::cudart)
