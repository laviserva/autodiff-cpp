cmake_minimum_required(VERSION 3.25)
project(micrograd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# Ejecutable principal
add_executable(autodiff_main src/main.cpp)

# Pruebas
if(BUILD_TESTING)
  enable_testing()
  add_executable(test_sanity tests/test_sanity.cpp)
  add_test(NAME SanityCheck COMMAND test_sanity)
endif()