# specify all source files for this test suite
set(TEST_SOURCES
        construction.cpp
        save.cpp
        get_knn_ids.cpp
        get_knn_points.cpp
)

# create google test
add_custom_test(knn "${TEST_SOURCES}")
