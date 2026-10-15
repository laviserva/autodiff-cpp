cmake_minimum_required(VERSION 3.25)
project(micrograd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# Librería central
add_library(autodiff_core INTERFACE)
target_include_directories(autodiff_core INTERFACE include)

# Ejecutable principal
add_executable(autodiff_main src/main.cpp)
target_link_libraries(autodiff_main PRIVATE autodiff_core)

# Pruebas
if(BUILD_TESTING)
  enable_testing()
  # Test de sanidad (Semana 1)
  add_executable(test_sanity tests/test_sanity.cpp)
  add_test(NAME SanityCheck COMMAND test_sanity)
  # Test de semántica de Tensor3D (Semana 2)
  add_executable(test_tensor tests/test_tensor.cpp)
  target_link_libraries(test_tensor PRIVATE autodiff_core)
  add_test(NAME TensorSemantics COMMAND test_tensor)
endif()