project(set10108-cw1)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

include_directories(../contrib)

find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_libraries(${OpenCL_LIBRARIES})


add_executable(cw1 main.cpp)
configure_file(calc-token-occurrences.cl calc-token-occurrences.cl COPYONLY)