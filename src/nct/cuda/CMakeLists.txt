
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  link_libraries(CUDA::cuda CUDA::cufft)
else()
  link_libraries(cuda)
  link_libraries("-framework Metal" "-framework Foundation")
endif()

aux_source_directory(. SRC)
add_library(nct_cuda ${SRC})
target_link_libraries(nct_cuda PRIVATE sfc)
