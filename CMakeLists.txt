cmake_minimum_required(VERSION 3.15)
project(index_hnsw_py LANGUAGES CXX)

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-mavx2 -mfma -march=native -O3 -ffast-math)

# 获取pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(pybind11)

# 源文件配置
file(GLOB_RECURSE SOURCE_FILES
    "src/*.cpp"
    "python/bindings.cpp"
)

# 创建共享库
add_library(index_hnsw_py SHARED ${SOURCE_FILES})

set_target_properties(index_hnsw_py PROPERTIES 
    PREFIX "" 
    SUFFIX ".so"
)

# 包含目录
target_include_directories(index_hnsw_py 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${pybind11_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(index_hnsw_py
    PRIVATE
    pybind11::module
)

# OpenMP支持
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(index_hnsw_py PRIVATE OpenMP::OpenMP_CXX)
endif()

# 设置输出目录
set_target_properties(index_hnsw_py PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    SUFFIX ".so"
)