
aux_source_directory(. SRC)
add_library(nct_data ${SRC})
target_link_libraries(nct_data PRIVATE sfc)
