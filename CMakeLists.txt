cmake_minimum_required(VERSION 3.10)
project(IndexFlatTest)

set(CMAKE_CXX_STANDARD 17)

# Add the source files
set(SOURCES
    main.cpp
    utils/Index.h
    utils/DistanceComputer.h
    utils/ResultHandler.h
)

# Add the executable
add_executable(IndexFlatTest ${SOURCES})

# Include directories if needed

include_directories(${CMAKE_SOURCE_DIR})
