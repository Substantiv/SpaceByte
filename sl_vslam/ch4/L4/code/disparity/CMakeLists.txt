cmake_minimum_required( VERSION 2.8 )
project( imageBasics )

# 添加c++ 14标准支持
set( CMAKE_CXX_FLAGS "-std=c++14" )

# 寻找Eigen库
find_package(Eigen3 REQUIRED)
include_directories( "/usr/include/eigen3" )

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 输出OpenCV头文件和库文件的路径
message(${OpenCV_INCLUDE_DIRS} )
message(${OpenCV_LIBS} )

# 寻找Pangolin库
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable( disparity disparity.cpp )
target_link_libraries( disparity ${OpenCV_LIBS} ${Pangolin_LIBRARIES} )


