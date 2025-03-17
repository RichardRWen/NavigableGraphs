set(TEST_FILES
    test_main.cpp
    greedy_set_cover.cpp
    unbounded_prune.cpp
    load_and_search.cpp
)

foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
endforeach()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/parlaylib/include)
include_directories(${CMAKE_SOURCE_DIR}/ParlayANN/algorithms)