cmake_minimum_required(VERSION 3.10)
project(IndexFlatTest)

set(CMAKE_CXX_STANDARD 17)

# 在CMakeLists.txt中添加以下内容
add_compile_options(-mavx2 -mfma)
add_compile_options(-march=native)
add_compile_options(-mavx512f)
add_compile_options(-mavx512vl)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add the source files
set(SOURCES
    main.cpp
    include/utils/Index.h
    include/utils/DistanceComputer.h
    include/utils/ResultHandler.h
    include/impl/HNSW.h
    include/impl/IndexHNSW.h
    include/impl/NodeDist.h
    src/HNSW.cpp
    src/IndexHNSW.cpp
    include/utils/RandomGenerator.h
    src/RandomGenerator.cpp
)

# Add the executable
add_executable(IndexFlatTest ${SOURCES})

# Include directories if needed
include_directories(${CMAKE_SOURCE_DIR})

# Link OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(IndexFlatTest PRIVATE OpenMP::OpenMP_CXX)
endif()
