cmake_minimum_required(VERSION 3.4)
project(quickrank)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -march=native -mtune=native -O3 -Wa,-q")

find_package(Boost REQUIRED COMPONENTS program_options system filesystem unit_test_framework)
set(Boost_USE_MULTITHREAD ON)  # enable multithreading

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(${Boost_INCLUDE_DIR})
include_directories(include)

# Put the binaries in the bin dir of the project
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

file(GLOB_RECURSE all_sources ${CMAKE_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE all_headers ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE all_tests ${CMAKE_SOURCE_DIR}/unit-tests/*.cc)

list(REMOVE_ITEM all_sources "${CMAKE_SOURCE_DIR}/src/quickscore.cc")

add_executable(quickrank ${all_headers} ${all_sources})
target_link_libraries(quickrank ${Boost_LIBRARIES})

list(REMOVE_ITEM all_sources "${CMAKE_SOURCE_DIR}/src/quicklearn.cc")

add_executable(unit-tests ${all_sources} ${all_tests})
target_link_libraries(unit-tests ${Boost_LIBRARIES})