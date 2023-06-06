cmake_minimum_required(VERSION 2.8)
project(chapter3)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)
find_package(OpenCV REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})



#SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)
# Eigen
include_directories("/usr/include/eigen3")
include_directories("/home/zhou/Pipeline_ws/src/chapter3")
include_directories(${PROJECT_SOURCE_DIR} 
                                        ${PROJECT_SOURCE_DIR}/include
                                        ${OpenCV_INCLUDE_DIRS}
                                         ${EIGEN3_INCLUDE_DIR}
                                        ${G2O_INCLUDE_DIRS}
                                        ${Sophus_INCLUDE_DIRS})


add_executable(cylinder_g2o cylinder.cpp)
target_link_libraries(cylinder_g2o
                                            g2o_core g2o_stuff g2o_types_slam3d ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY}
                                             ${OpenCV_LIBS})

add_executable(autodiff autodiff.cpp)
target_link_libraries(autodiff
                                             ${OpenCV_LIBS} )