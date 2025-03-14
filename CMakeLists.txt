cmake_minimum_required(VERSION 3.10)
project(IndexFlatTest)

set(CMAKE_CXX_STANDARD 17)

# Add the source files
set(SOURCES
    main.cpp
    utils/Index.h
    utils/DistanceComputer.h
    utils/ResultHandler.h
    impl/HNSW.h
    impl/IndexHNSW.h
    impl/NodeDist.h
    impl/HNSW.cpp
    impl/IndexHNSW.cpp
    utils/RandomGenerator.h
    utils/RandomGenerator.cpp
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
