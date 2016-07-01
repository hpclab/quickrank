# compile with CMake.
# G++ COMPILER OPTION (on Mac OS X, brew): -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-5
# GCC COMPILER OPTION (on Mac OS X, brew): -DCMAKE_C_COMPILER=/usr/local/bin/gcc-5

cmake_minimum_required(VERSION 3.4)
project(QUICKRANK)

# require support of c++11
set_property(GLOBAL PROPERTY CXX_STANDARD 11)
set_property(GLOBAL PROPERTY CXX_STANDARD_REQUIRED ON)

# require support of openmp
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Compiler flags (todo: fix release vs. debug)
# SET(CMAKE_BUILD_TYPE Release CACHE STRING "Release is the default build type")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -march=native -mtune=native -O3") # for franco  -Wa,-q
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -g") # for franco  -Wa,-q
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -march=native -mtune=native -O3") # for franco  -Wa,-q

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -march=native -mtune=native -O3 -Wa,-q")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -O0 -g -Wa,-q")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -march=native -mtune=native -O3 -Wa,-q")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -O0 -g -Wa,-q")
endif()

# Binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# -----------------------------------------
# boost to be removed
find_package(Boost REQUIRED COMPONENTS program_options system filesystem unit_test_framework)
set(Boost_USE_MULTITHREAD ON)  # enable multithreading
include_directories(${Boost_INCLUDE_DIR})
# -----------------------------------------

# include files path
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories("lib")

# source files path, excluding main files
set(QUICKLEARN_MAIN "${CMAKE_SOURCE_DIR}/src/quicklearn.cc")
set(QUICKSCORE_MAIN "${CMAKE_SOURCE_DIR}/src/quickscore.cc")
file(GLOB_RECURSE all_sources ${CMAKE_SOURCE_DIR}/src/*.cc)
list(REMOVE_ITEM all_sources ${QUICKLEARN_MAIN})
list(REMOVE_ITEM all_sources ${QUICKSCORE_MAIN})
file(GLOB_RECURSE all_headers ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE unit_tests_sources ${CMAKE_SOURCE_DIR}/catch-unit-tests/*.cc)

# ---------------------------------
# quicklearn target
add_executable(quicklearn ${all_headers} ${all_sources} ${QUICKLEARN_MAIN})

# ---------------------------------
# quickscore target

# ---------------------------------
# download test data
add_custom_command( OUTPUT ${CMAKE_SOURCE_DIR}/quickranktestdata
    COMMAND git clone http://git.hpc.isti.cnr.it/quickrank/quickranktestdata.git ${CMAKE_SOURCE_DIR}/quickranktestdata
    COMMENT "Cloning Quickrank small datasets repo"
    )
add_custom_target( quickranktestdata DEPENDS ${CMAKE_SOURCE_DIR}/quickranktestdata )

# ---------------------------------
# unit-test target
add_executable(unit-tests ${all_headers} ${all_sources} ${unit_tests_sources})
add_dependencies(unit-tests quickranktestdata)

# ---------------------------------
# generate documentation
add_custom_target( doc
    COMMAND doxygen quickrank.doxygen
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/documentation
    COMMENT "Creating the documentation with doxygen"
    )




