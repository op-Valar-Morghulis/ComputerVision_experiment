cmake_minimum_required(VERSION 3.9)
project(computer_vision_experiment1)

set(CMAKE_CXX_STANDARD 11)

#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)

add_executable(computer_vision_experiment1 alpha合成.cpp)

target_link_libraries(computer_vision_experiment1 ${OpenCV_LIBS})
