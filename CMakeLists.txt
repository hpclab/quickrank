# QuickRank - CMakeLists File.
# GNU C++ COMPILER OPTION (on Mac OS X, brew): -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-5
# BUILD TYPE: -DCMAKE_BUILD_TYPE=[Release|Debug]

cmake_minimum_required(VERSION 2.8.12)
project(QUICKRANK CXX)

# require support of C++11
set_property(GLOBAL PROPERTY CXX_STANDARD 11)
set_property(GLOBAL PROPERTY CXX_STANDARD_REQUIRED ON)

# handling build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# require support of OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# explicitly set default CMAKE_CXX_FLAGS_RELEASE options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# explicitly set default CMAKE_CXX_FLAGS_DEBUG options
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# Compiler flags
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -march=native -mtune=native -Wa,-q")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -O0 -Wa,-q")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -march=native -mtune=native")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -O0")
endif()

# Binary output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# include header and lib paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(lib)

# source files path, excluding main files
set(QUICKLEARN_MAIN "${CMAKE_SOURCE_DIR}/src/quicklearn.cc")
set(QUICKSCORE_MAIN "${CMAKE_SOURCE_DIR}/src/quickscore.cc")
set(RANKER_CC "${CMAKE_SOURCE_DIR}/src/scoring/ranker.cc")
file(GLOB_RECURSE all_sources ${CMAKE_SOURCE_DIR}/src/*.cc)
list(REMOVE_ITEM all_sources ${QUICKLEARN_MAIN})
list(REMOVE_ITEM all_sources ${QUICKSCORE_MAIN})
list(REMOVE_ITEM all_sources ${RANKER_CC})

file(GLOB_RECURSE unit_tests_sources ${CMAKE_SOURCE_DIR}/catch-unit-tests/*.cc)

# external libraries
file(GLOB_RECURSE pugixml_sources ${CMAKE_SOURCE_DIR}/lib/pugixml/src/*.cpp)
add_library(pugixml STATIC ${pugixml_sources})
add_library(quickrank_common STATIC ${all_sources})
target_link_libraries(quickrank_common pugixml)

# managing QuickRank headers and libraries
file(GLOB_RECURSE all_headers
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/lib/pugixml/src/*.hpp
        ${CMAKE_SOURCE_DIR}/lib/catch/include/catch.hpp
        ${CMAKE_SOURCE_DIR}/lib/paramsmap/*.h)

# ---------------------------------
# quicklearn target
add_executable(quicklearn ${all_headers} ${QUICKLEARN_MAIN})
target_link_libraries(quicklearn quickrank_common)

# ---------------------------------
# quickscore target
add_executable(quickscore EXCLUDE_FROM_ALL ${all_headers} ${RANKER_CC} ${QUICKSCORE_MAIN})
target_link_libraries(quickscore quickrank_common)

# ---------------------------------
# download test data for unit-testing
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/quickranktestdata
    COMMAND git clone http://git.hpc.isti.cnr.it/quickrank/quickranktestdata.git ${CMAKE_SOURCE_DIR}/quickranktestdata
    COMMENT "Cloning QuickRank small datasets repo"
)
add_custom_target(quickranktestdata DEPENDS ${CMAKE_SOURCE_DIR}/quickranktestdata)

# ---------------------------------
# unit-test target
add_executable(unit-tests EXCLUDE_FROM_ALL ${all_headers} ${unit_tests_sources})
add_dependencies(unit-tests quickranktestdata)
target_link_libraries(unit-tests quickrank_common)

# ---------------------------------
# generate documentation
add_custom_target(doc
    COMMAND doxygen quickrank.doxygen
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/api-documentation
    COMMENT "Creating the documentation with doxygen"
)
