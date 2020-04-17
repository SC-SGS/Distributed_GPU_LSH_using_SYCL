### based on https://scicomp.stackexchange.com/questions/8516/any-recommendations-for-unit-testing-frameworks-compatible-with-code-libraries-t ###

# easily create mpi test cases
function(add_mpi_test name test_files num_mpi_procs)
    # add new test executable with the static main and the provided source test files
    add_executable(${name} ${CMAKE_SOURCE_DIR}/test/main.cpp ${test_files})
    # add enable death test cmake flag
    if(ENABLE_DEATH_TESTS)
        target_compile_definitions(${name} PRIVATE ENABLE_DEATH_TESTING=1)
    else()
        target_compile_definitions(${name} PRIVATE ENABLE_DEATH_TESTING=0)
    endif()
    # link test against own library and gtest
    target_link_libraries(${name} ${LIBRARY_NAME} gtest)
    # add MPI parameter for the number of used processors
    set(test_parameters ${MPIEXEC_NUMPROC_FLAG} ${num_mpi_procs} "./${name}")
    # add test to run with MPI
    add_test(NAME ${name} COMMAND ${MPIEXEC} ${test_parameters})
endfunction(add_mpi_test)

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