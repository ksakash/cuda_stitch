cmake_minimum_required(VERSION 2.8)
project(cuda_stitch)
find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories (${EIGEN3_INCLUDE_DIRS})

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${Eigen3_LIBS})
