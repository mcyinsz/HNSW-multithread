#  pragma once

#include <random>

class RandomGenerator {

    private:
        std::mt19937 mt;

    public:

        // construct function
        explicit RandomGenerator(int64_t seed = 42);
        
        // random positive integer
        int rand_int();

        // generate random integer in [0, max)
        int rand_int(int max);

        // random float between [0, 1]
        float rand_float();
};
