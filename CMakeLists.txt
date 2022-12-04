cmake_minimum_required(VERSION 3.0.0)
project(seg_demo VERSION 0.1.0)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/third_party/ncnn/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
)

file(GLOB_RECURSE SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

add_executable(seg_demo ${SRC_LIST})
target_link_libraries(seg_demo ${OpenCV_LIBS} ncnn)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)