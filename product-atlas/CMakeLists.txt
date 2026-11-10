cmake_minimum_required(VERSION 3.16)
project(product_atlas VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(product_atlas src/product_atlas.cpp)
target_include_directories(product_atlas PUBLIC include)
target_compile_options(product_atlas PRIVATE -Wall -Wextra -Wpedantic)

add_executable(product_atlas_bench src/benchmark.cpp)
target_link_libraries(product_atlas_bench PRIVATE product_atlas)
target_compile_options(product_atlas_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(product_atlas_tests tests/test_product_atlas.cpp)
target_link_libraries(product_atlas_tests PRIVATE product_atlas)
target_compile_options(product_atlas_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME product_atlas_tests COMMAND product_atlas_tests)
