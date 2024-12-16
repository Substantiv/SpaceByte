cmake_minimum_required(VERSION 3.0.2)
project(IRIS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(IRIS main.cpp
        IRIS.cpp
        IRIS.h)

find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

# 包含目录
include_directories(
        ${EIGEN3_INCLUDE_DIRS}
        ${OSQP_INCLUDE_DIRS}
)

# 目标链接库
target_link_libraries(IRIS PRIVATE
        Eigen3::Eigen
        osqp::osqp
        OsqpEigen::OsqpEigen)


target_include_directories(IRIS PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
        ${OSQP_INCLUDE_DIRS}
)
