
aux_source_directory(. SRC)
add_library(nct_math ${SRC})
target_link_libraries(nct_math sfc nct_cuda)
