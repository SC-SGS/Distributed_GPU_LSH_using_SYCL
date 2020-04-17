# specify all source files for this test suite
set(TEST_SOURCES
    dummy.cpp
)

# create google test with MPI support
add_mpi_test(dummy "${TEST_SOURCES}" 1)
