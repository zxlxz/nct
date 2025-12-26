
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  link_directories(CUDA::cudart CUDA::cufft)
elseif(EXISTS /opt/cuda)
  include_directories(/opt/cuda/include)
  link_directories(/opt/cuda/lib)
endif()

aux_source_directory(. SRC)
add_library(nct_cuda ${SRC})
target_link_libraries(nct_cuda PRIVATE sfc)
