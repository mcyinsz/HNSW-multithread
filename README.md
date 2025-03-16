# HNSW Implementation with Enhanced Features

This project introduces an optimized implementation of the Hierarchical Navigable Small World (HNSW) algorithm, featuring significant improvements in both performance and scalability. The key enhancements include:

1. **Parallel Build Method**: The implementation now supports parallel graph construction, enabling faster processing of large datasets.

2. **AVX Optimization**: Leveraging Advanced Vector Extensions (AVX) instructions for improved computational efficiency and faster vector operations.

The codebase is derived from the FAISS library (<https://github.com/facebookresearch/faiss>), but has been significantly streamlined and simplified. 

## build

* generate Cmakefile

``` bash
mkdir build
cd build
cmake ..
```

* compile

``` bash
make
```

## Performance

| dim | n vectors | build time (s)|
|:---:|:---:|:---:|
|128|131072|863.373|
|128|10000|35.1514|
|128|5000|15.4052|
|128|1000|1.1486|
|1280|1000|2.0724|
|1280|10000|39.813|

* optimization: using `avx2` ISA to accelerate distance computing

| dim | n vectors | build time (s)|
|:---:|:---:|:---:|
|128|100000|160|
|128|10000|10|
|128|5000|3.2|
|128|1000|0.3|

* optimization 2: using `avx512` ISA to accelerate distance computing
* optimization 3: add `O3` optimization in `CMakeList.txt`

| dim | n vectors | build time (s)|
|:---:|:---:|:---:|
|128|1310720|<250|
|128|100000|<15|
|128|10000|<0.9|
|128|5000|<0.5|
|128|1000|<0.03|
|1280|100000|<80|
|1280|10000|<6|

## Recall

* degree limit for level > 0: 64
* degree limit for level = 0: 32
* build beam search width: 40
* dataset size: 500k

![](pics\test_IndexHNSW_recall_dim128_size500000_normal1.csv.png)
