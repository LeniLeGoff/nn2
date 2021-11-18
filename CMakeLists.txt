cmake_minimum_required(VERSION 3.10)
project(nn2)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

message("Load nn2 module")

set(WITH_NN2 ON CACHE INTERNAL "WITH_NN2")

add_compile_options("$<$<CONFIG:RELEASE>:-O3>")

# DEPENDENCIES
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Eigen3 REQUIRED)

# NN2 INTERFACE LIBRARY
file(GLOB HEADERS "*.hpp" "*.h")
add_library(NN2 INTERFACE)

target_link_libraries(NN2 INTERFACE Eigen3::Eigen Boost::serialization)
target_include_directories(NN2 SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_compile_definitions(NN2 INTERFACE -DEIGEN3_ENABLED ${EIGEN3_DEFINITIONS})

# TESTS SECTION
if(TEST_CPPN)
    find_package(OpenCV REQUIRED)

    add_executable(test_cppn tests/test_cppn.cpp)
    target_link_libraries(test_cppn NN2 ${OpenCV_LIBRARIES})
    target_include_directories(test_cppn PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_cppn_serialization tests/test_cppn_serialization.cpp)
    target_link_libraries(test_cppn_serialization NN2 ${OpenCV_LIBRARIES})
    target_include_directories(test_cppn_serialization PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_cpg tests/test_cpg.cpp)
    target_link_libraries(test_cpg NN2 ${OpenCV_LIBRARIES})
    target_include_directories(test_cpg PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_cpg2 tests/test_cpg2.cpp)
    target_link_libraries(test_cpg2 NN2 ${OpenCV_LIBRARIES})
    target_include_directories(test_cpg2 PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

endif()


# INSTALL SECTION
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/NN2)

#set_target_properties(NN2 PROPERTIES PUBLIC_HEADER "${HEADERS}")
set_target_properties(NN2 PROPERTIES EXPORT_NAME NN2)


install(TARGETS NN2
        EXPORT NN2-targets
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nn2"
        )

install(FILES ${HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nn2")

install(EXPORT NN2-targets
        FILE NN2Targets.cmake
        NAMESPACE NN2::
        DESTINATION ${INSTALL_CONFIGDIR})

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/NN2Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/NN2Config.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/NN2Config.cmake
        DESTINATION ${INSTALL_CONFIGDIR})
