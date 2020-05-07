# specify all source files for this test suite
set(TEST_SOURCES
        construction.cpp
        copy_to_different_layout.cpp
        indexing.cpp
)

# create google test
add_custom_test(data "${TEST_SOURCES}")
