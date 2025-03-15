# Project overview

## project tree

``` bash
├── CMakeLists.txt
├── README.md
├── impl
│   ├── HNSW.cpp
│   ├── HNSW.h
│   ├── IndexHNSW.cpp
│   ├── IndexHNSW.h
│   └── NodeDist.h
├── main.cpp
└── utils
    ├── DistanceComputer.cpp
    ├── DistanceComputer.h
    ├── Index.h
    ├── RandomGenerator.cpp
    ├── RandomGenerator.h
    ├── ResultHandler.h
    ├── VisitedTable.h
    └── constants.h
```

## functional files

* `Index.h`: index interface including `add`, `search`. also contains the defination of `IndexFlat`
* `HNSW.cpp`: the graph structure and operations for HNSW (levels, links, add link logic, search logic ...)
* `IndexHNSW.cpp`: interconnect `Index` interface and HNSW graph structure
* `DistanceComputer.cpp`: for efficient computing of vector distances (based on `AVX` extension)
* `RandomGenerator.cpp`: for generating random `int`, `float` numbers
* `ResultHandler.cpp`: for storing kNN search results
* `VisitedTable.h`: for record which graph nodes have been reached

