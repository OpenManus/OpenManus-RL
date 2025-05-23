cmake_minimum_required(VERSION 3.15)

project(py_ai_sdk CXX)

set(CMAKE_BUILD_TYPE Release)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /std:c++17")
else(GCC or Clang)
    set(CMAKE_CXX_STANDARD 17)
endif(MSVC)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# set(CMAKE_PREFIX_PATH "<Python-path>")
# find_package(Python <Python-version> EXACT COMPONENTS Interpreter Development REQUIRED)
set(pybind11_DIR
    "/home/piano/mambaforge/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

file(GLOB_RECURSE jsoncpp "jsoncpp/*.cpp")
pybind11_add_module(ai_client py_ai_sdk.cpp ${jsoncpp})
