cmake_minimum_required(VERSION 3.10)
project(detect_laser_pointers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)


# Add executable
add_executable(detect_laser_pointers detect_laser_points.cpp)  # Replace main.cpp with your actual filename

target_include_directories(detect_laser_pointers PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(detect_laser_pointers
    PRIVATE
    ${OpenCV_LIBS}
    )
