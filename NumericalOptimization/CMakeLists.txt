cmake_minimum_required(VERSION 3.30)
project(NumericalOptimization)

set(CMAKE_CXX_STANDARD 20)

add_executable(NumericalOptimization main.cpp
        steepestGD.h)
