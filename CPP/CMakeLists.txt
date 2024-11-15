# 设置项目的最低CMake版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(MyProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 查找 Eigen 库
find_package(Eigen3 3.3 REQUIRED)

# 如果找不到Eigen, 则显示错误
if (NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 not found! Please install Eigen.")
endif()

# 添加源文件
add_executable(EigenExecutable EigenExample.cpp)
add_executable(KeyWordExecutable KeyWordExample.cpp)

# 包含 Eigen 头文件目录
target_include_directories(EigenExecutable PRIVATE ${EIGEN3_INCLUDE_DIRS})


