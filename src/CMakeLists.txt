find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_libraries(CUDA::cudart)


file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/**.c[cu]")
add_library(nct ${SRC})

target_compile_features(nct PUBLIC cxx_std_23)
target_include_directories(nct PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(nct PUBLIC sfc)
