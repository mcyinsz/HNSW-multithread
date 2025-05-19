cmake_minimum_required(VERSION 3.15)
project(index_hnsw_py LANGUAGES CXX)

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-mavx2 -mfma -march=native -O3 -ffast-math)

# 首先查找完整的 Python 开发环境
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# 设置 pybind11 查找路径
set(pybind11_DIR "$ENV{CONDA_PREFIX}/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

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
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# 包含目录
target_include_directories(index_hnsw_py 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Python_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(index_hnsw_py
    PRIVATE
    pybind11::module
    ${Python_LIBRARIES}
)

# OpenMP支持
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(index_hnsw_py PRIVATE OpenMP::OpenMP_CXX)
endif()
