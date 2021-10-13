cmake_minimum_required(VERSION 3.0)
project(nn2)
add_compile_options("$<$<CONFIG:RELEASE>:-O3>")

message("Load nn2 module")

set(WITH_NN2 ON CACHE INTERNAL "WITH_NN2")

find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Eigen3 REQUIRED)

file(GLOB HEADERS "*.hpp" "*.h")
add_library(NN2 INTERFACE ${HEADERS})
set_target_properties(NN2 PROPERTIES PUBLIC_HEADER "${HEADERS}")

target_link_libraries(NN2 INTERFACE Eigen3::Eigen Boost::serialization ${OpenCV_LIBS})
target_include_directories(NN2 INTERFACE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
target_compile_definitions(NN2 INTERFACE -DEIGEN3_ENABLED ${EIGEN3_DEFINITIONS})


if(TEST_CPPN)
    find_package(OpenCV REQUIRED)

    add_executable(test_cppn tests/test_cppn.cpp)
    target_link_libraries(test_cppn NN2 ${OpenCV_LIBRARIES})
    target_include_directories(test_cppn PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_cppn_serialization tests/test_cppn_serialization.cpp)
    target_link_libraries(test_cppn_serialization NN2)
    target_link_libraries(test_cppn_serialization ${OpenCV_LIBRARIES})
    target_include_directories(test_cppn_serialization PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

endif()

#install(DIRECTORY . DESTINATION include/nn2/ FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" )
install(TARGETS NN2 PUBLIC_HEADER DESTINATION "include/nn2")

