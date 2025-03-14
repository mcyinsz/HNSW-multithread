# Distance Computer

## initialization

`qdis = new DistanceComputer(std::vector<float> vectors, int d);`

## properties

* `int` type vector dimension `d`
* `std::vector<float>&` type `storage`
* `float*` type query vector pointer `q`

## methods

* `operator` -> `float`
    * function: calculate the distance between the query vector and the `i`th vector in storage
    * call: `(*qdis)(int index)`
* `symmetric_dis` -> `float`
    * function: calculate the distance between two vectors in storage
    * call: `symmetric_dis(int idx_a, int idx_b)`
* `set_query` -> `None`
    * function: set a vector to be the query vector
    * call: `qdis->set_query(x.data() + i * d);`

## AVX supports