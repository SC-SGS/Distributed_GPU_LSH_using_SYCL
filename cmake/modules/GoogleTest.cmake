### based on https://github.com/google/googletest/blob/master/googletest/README.md ###

# Download and unpack googletest at configure time
configure_file(${CMAKE_SOURCE_DIR}/third_party/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/third_party/googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/googletest-download )

if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/googletest-download )

if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest-src
                 ${CMAKE_SOURCE_DIR}/third_party/googletest-build
                 EXCLUDE_FROM_ALL)
