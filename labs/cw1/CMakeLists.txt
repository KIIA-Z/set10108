project(set10108-cw1)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

include_directories(../contrib)

#copy over txt files in data folder
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/dataset")
set(DEST_DIR "${CMAKE_BINARY_DIR}")
file(COPY ${SOURCE_DIR} DESTINATION ${DEST_DIR})

find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_libraries(${OpenCL_LIBRARIES})


add_executable(cw1 main.cpp)
configure_file(calc-token-occurrences.cl calc-token-occurrences.cl COPYONLY)