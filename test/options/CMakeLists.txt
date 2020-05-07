# specify all source files for this test suite
set(TEST_SOURCES
        factory.cpp
        options.cpp
)

# create google test
add_custom_test(options "${TEST_SOURCES}")
