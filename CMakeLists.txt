cmake_minimum_required(VERSION 3.0)

add_definitions(-DEIGEN3_ENABLED)

add_executable(bench_nn bench_nn.cpp)
target_include_directories(bench_nn PUBLIC "/usr/include/eigen3")

install(DIRECTORY *.hpp *.h DESTINATION include)

