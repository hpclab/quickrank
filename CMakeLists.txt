cmake_minimum_required(VERSION 3.4)
project(quicklearn)

#set( CMAKE_VERBOSE_MAKEFILE on )

# set compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -march=native -mtune=native -O3 -Wa,-q")

# Put the binaries in the bin dir of the project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

find_package(Boost REQUIRED COMPONENTS program_options system filesystem unit_test_framework)
set(Boost_USE_MULTITHREAD ON)  # enable multithreading

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(${Boost_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE all_sources ${CMAKE_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE all_headers ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE unit_tests_sources ${CMAKE_SOURCE_DIR}/unit-tests/*.cc)

set(quicklearn_sources ${all_sources})
list(REMOVE_ITEM quicklearn_sources "${CMAKE_SOURCE_DIR}/src/quickscore.cc")

set(quickscore_sources ${all_sources})
list(REMOVE_ITEM quickscore_sources "${CMAKE_SOURCE_DIR}/src/quicklearn.cc")

set(all_sources_wo_exec ${all_sources})
list(REMOVE_ITEM all_sources_wo_exec "${CMAKE_SOURCE_DIR}/src/quicklearn.cc
                                      ${CMAKE_SOURCE_DIR}/src/quickscore.cc")

# right now quickscore is not compiled
add_executable(quicklearn ${all_headers} ${quicklearn_sources})
target_link_libraries(quicklearn ${Boost_LIBRARIES})

add_executable(unit-tests ${all_sources_wo_exec} ${unit_tests})
target_link_libraries(unit-tests ${Boost_LIBRARIES})