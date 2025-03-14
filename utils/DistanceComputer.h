#ifndef DISTANCE_COMPUTER_H
#define DISTANCE_COMPUTER_H
#pragma once
#include <vector>

class DistanceComputer {
    public:
        virtual float symmetric_dis(int idx_a, int idx_b) = 0;
        virtual void set_query(const float* x) = 0;
        virtual void set_query_storage(int idx) = 0;
        virtual float operator()(int index) = 0;
        virtual ~DistanceComputer() = default;
};
    
#endif // DISTANCE_COMPUTER_H