cmake_minimum_required(VERSION 3.10)
project(cRoboticsController)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
include(FindcRoboticsEnv)

# Conda 경로를 CMAKE_PREFIX_PATH에 추가 (서드파티 라이브러리 검색용)
list(APPEND CMAKE_PREFIX_PATH "${CONDA_ENV_PATH}")
if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH "${CONDA_ENV_PATH}/Library")
endif()

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(eigenpy REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${eigenpy_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
    src/controller.cpp
)

target_link_libraries(${PROJECT_NAME} 
    Eigen3::Eigen
    pinocchio::pinocchio
)

add_library(${PROJECT_NAME}_wrapper_cpp SHARED
    src/bindings.cpp
)

target_link_libraries(${PROJECT_NAME}_wrapper_cpp
    ${PROJECT_NAME}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    Python3::Python
    eigenpy::eigenpy
)

set_target_properties(${PROJECT_NAME}_wrapper_cpp PROPERTIES
    OUTPUT_NAME "${PROJECT_NAME}_wrapper_cpp"
    PREFIX ""
)

if(WIN32)
    set_target_properties(${PROJECT_NAME}_wrapper_cpp PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(${PROJECT_NAME}_wrapper_cpp PROPERTIES SUFFIX ".so")
endif()

if(NOT WIN32)
    set(CONDA_LIB_PATH "${CONDA_ENV_PATH}/lib")

    set_target_properties(${PROJECT_NAME}_wrapper_cpp PROPERTIES
        BUILD_RPATH   "${CONDA_LIB_PATH}"
        INSTALL_RPATH "${CONDA_LIB_PATH}"
        SKIP_BUILD_RPATH FALSE
        SKIP_INSTALL_RPATH FALSE
    )
endif()