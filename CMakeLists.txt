cmake_minimum_required(VERSION 3.10)
project(face_recognition_video)

set(CMAKE_CXX_STANDARD 11)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Adding definitions
add_definitions(-DUSING_OPENCL)

find_package( OpenCV REQUIRED )

add_executable(${PROJECT_NAME} main.cpp face_recognition_video.cpp face_recognition_video.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


