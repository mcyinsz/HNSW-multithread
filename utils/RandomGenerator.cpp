#include <RandomGenerator.h>

RandomGenerator::RandomGenerator(int64_t seed) : mt((unsigned int)seed) {}

int RandomGenerator::rand_int() {
    return mt() & 0x7fffffff;
}


int RandomGenerator::rand_int(int max) {
    return mt() % max;
}

float RandomGenerator::rand_float() {
    return mt() / float(mt.max());
}
