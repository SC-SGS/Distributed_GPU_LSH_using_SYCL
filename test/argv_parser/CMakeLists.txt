# specify all source files for this test suite
set(TEST_SOURCES
        construction.cpp
        has_argv.cpp
        argv_as.cpp
        description.cpp
)

# create google test
add_custom_test(argv_parser "${TEST_SOURCES}")
