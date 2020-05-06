### based on https://scicomp.stackexchange.com/questions/8516/any-recommendations-for-unit-testing-frameworks-compatible-with-code-libraries-t ###

# easily create test cases
function(add_custom_test name test_files)
    # add new test executable with the static main and the provided source test files
    add_executable(${name} ${CMAKE_SOURCE_DIR}/test/test_main.cpp ${test_files})
    # link against gtest
    target_link_libraries(${name} PRIVATE gtest)
    # add test
    add_test(NAME ${name} COMMAND ${name})
endfunction(add_custom_test)

### based on https://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory ###
function(list_subdirectories result curdir)
    file(GLOB children RELATIVE ${curdir} ${curdir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${curdir}/${child})
            list(APPEND dirlist ${child})
        endif()
    endforeach()
    set(${result} ${dirlist} PARENT_SCOPE)
endfunction(list_subdirectories)

# add all test suites
list_subdirectories(TEST_DIRS ${PROJECT_SOURCE_DIR}/test)

list(LENGTH TEST_DIRS NUM_TEST_DIRS)
message(STATUS "Added ${NUM_TEST_DIRS} test directories")
foreach(test_dir ${TEST_DIRS})
    add_subdirectory(${test_dir})
endforeach()