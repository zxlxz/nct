
file(GLOB_RECURSE CC_SRC "${CMAKE_CURRENT_LIST_DIR}/*.cc")
file(GLOB_RECURSE CU_SRC "${CMAKE_CURRENT_LIST_DIR}/*.cu")

add_library(nct_recon ${CC_SRC} ${CU_SRC})
target_link_libraries(nct_recon PRIVATE sfc)
