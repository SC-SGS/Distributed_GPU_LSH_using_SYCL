# specify all source files for this test suite
set(TEST_SOURCES
        construction.cpp
)

# create google test with MPI support
add_custom_test(data "${TEST_SOURCES}")
