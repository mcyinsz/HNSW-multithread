# HNSW implementation

Enabling parallel build method. 

The source code is based on `FAISS` (<https://github.com/facebookresearch/faiss>), but more simplified for further modification.

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