cmake_minimum_required(VERSION 3.0)

message("Load nn2 module")

set(WITH_NN2 ON CACHE INTERNAL "WITH_NN2")

add_definitions(-DEIGEN3_ENABLED)

add_executable(bench_nn bench_nn.cpp)
target_include_directories(bench_nn PUBLIC "/usr/include/eigen3")

install(DIRECTORY . DESTINATION include/nn2/ FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" )

