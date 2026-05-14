cmake_minimum_required(VERSION 3.18)
project(resgrid_plotter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(
  resgrid_plotter
  src/plotter_management.cu
  src/plotter_pipeline.cu
  src/plotter_posix_mq.cpp
  src/plotter_assembler.cu
  src/plotter_raster.cu
  src/plotter_encoder.cu
  src/plotter_capturer.cu
)
target_include_directories(
  resgrid_plotter
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(resgrid_plotter PRIVATE CUDA::cudart)
if(UNIX AND NOT APPLE)
  target_link_libraries(resgrid_plotter PRIVATE rt)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set_property(TARGET resgrid_plotter PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)
endif()
