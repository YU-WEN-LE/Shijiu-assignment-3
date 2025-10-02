cmake_minimum_required(VERSION 3.10)
project(main)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")  


find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio)
find_package(Glog REQUIRED)
find_package(Eigen3 REQUIRED)


if(NOT Ceres_FOUND)
    message(FATAL_ERROR "Ceres Solver not found. Please install Ceres.")
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV.")
endif()


include_directories(
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)


add_executable(main main.cpp)


target_link_libraries(main
    PRIVATE
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
)


install(TARGETS main
    RUNTIME DESTINATION bin
)
    
