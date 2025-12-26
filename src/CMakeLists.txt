

set(CMAKE_CXX_STANDARD 23)
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  enable_language(CUDA)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(nct/math)
add_subdirectory(nct/cuda)

add_subdirectory(nct/data)
add_subdirectory(nct/corr)
add_subdirectory(nct/recon)
add_subdirectory(nct/params)

add_library(nct INTERFACE)
target_link_libraries(nct INTERFACE nct_math nct_cuda)
target_link_libraries(nct INTERFACE nct_data nct_corr nct_recon)
target_include_directories(nct INTERFACE ${CMAKE_CURRENT_LIST_DIR})
