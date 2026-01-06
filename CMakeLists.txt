cmake_minimum_required(VERSION 3.15)
project(helicopter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED CONFIG)
find_package(Eigen3 3.3 QUIET CONFIG)

if(NOT Eigen3_FOUND)
    set(EIGEN3_INCLUDE_DIR "${CONDA_PREFIX}/include/eigen3")
    message(STATUS "Eigen3 config not found, falling back to Conda path: ${EIGEN3_INCLUDE_DIR}")
else()
    message(STATUS "Found Eigen3 via CMake config")
endif()

set(module_name helicopter_cpp)
set(SRC_FILES cpp/src/bindings.cpp)

pybind11_add_module(${module_name} MODULE ${SRC_FILES})

set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/helicopter"
)

target_include_directories(${module_name} PRIVATE
        "cpp/src"
)

if(Eigen3_FOUND)
    target_link_libraries(${module_name} PRIVATE Eigen3::Eigen)
else()
    target_include_directories(${module_name} PRIVATE "${EIGEN3_INCLUDE_DIR}")
endif()

if(MSVC)
    target_compile_options(${module_name} PRIVATE /W4)
else()
    target_compile_options(${module_name} PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS ${module_name} LIBRARY DESTINATION helicopter)

find_program(STUBGEN_EXE pybind11-stubgen)
if(STUBGEN_EXE)
    add_custom_command(TARGET ${module_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH="${CMAKE_CURRENT_SOURCE_DIR}/helicopter"
            ${STUBGEN_EXE} ${module_name} -o "${CMAKE_CURRENT_SOURCE_DIR}/helicopter"
            COMMENT "Updating IDE autocomplete stubs..."
    )
endif()