# HNSW implementation

enabling parallel build method

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

* using `std::vector<float>` to add vectors

| dim | n vectors | build time (s)|
|:---:|:---:|:---:|
|128|10000|35.1514|
|128|5000|15.4052|
|128|1000|1.1486|
|1280|1000|2.0724|
|1280|10000|39.813|