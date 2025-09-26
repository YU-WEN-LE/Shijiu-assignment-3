cmake_minimum_required(VERSION 3.10)
project(main)

# 设置C++标准版本，确保兼容性和现代特性支持
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")  # 启用优化和警告

# 寻找依赖库
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio)
find_package(Glog REQUIRED)
find_package(Eigen3 REQUIRED)

# 检查依赖是否找到
if(NOT Ceres_FOUND)
    message(FATAL_ERROR "Ceres Solver not found. Please install Ceres.")
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV.")
endif()

# 包含头文件目录
include_directories(
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# 生成可执行文件
add_executable(main main.cpp)

# 链接所有依赖库
target_link_libraries(main
    PRIVATE
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
)

# 安装目标（可选）
install(TARGETS main
    RUNTIME DESTINATION bin
)
    
