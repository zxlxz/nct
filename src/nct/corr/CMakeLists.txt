
aux_source_directory(. SRC)
add_library(nct_corr ${SRC})
target_link_libraries(nct_corr PRIVATE sfc)
