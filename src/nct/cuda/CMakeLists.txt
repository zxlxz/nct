
find_package(CUDAToolkit)
enable_language(CUDA)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_directories(CUDA::cudart CUDA::cufft)

aux_source_directory(. SRC)
add_library(nct_cuda ${SRC})
