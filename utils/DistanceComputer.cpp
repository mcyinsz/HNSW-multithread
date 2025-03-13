#include <vector>
#include <cmath>
#include "utils/DistanceComputer.h"

class GenericDistanceComputerIP : public DistanceComputer {
    int d;  // 向量维度
    const std::vector<float>& storage;  // 连续内存存储的向量数据
    const float* q;  // 查询向量的指针

public:
    GenericDistanceComputerIP(const std::vector<float>& storage, int d) 
        : storage(storage), d(d), q(nullptr) {}

    float operator()(int index) override {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += q[i] * storage[index * d + i];
        }
        return -sum;
    }

    float symmetric_dis(int idx_a, int idx_b) override {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += storage[idx_a * d + i] * storage[idx_b * d + i];
        }
        return -sum;
    }

    void set_query(const float* x) override {
        q = x;
    }

    void set_query_storage(int idx) override {
        q = &storage[idx * d];
    }
};

class GenericDistanceComputerL2 : public DistanceComputer {
    int d;  // 向量维度
    const std::vector<float>& storage;  // 连续内存存储的向量数据
    const float* q;  // 查询向量的指针

public:
    GenericDistanceComputerL2(const std::vector<float>& storage, int d) 
        : storage(storage), d(d), q(nullptr) {}

    float operator()(int index) override {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = q[i] - storage[index * d + i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    float symmetric_dis(int idx_a, int idx_b) override {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = storage[idx_a * d + i] - storage[idx_b * d + i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    void set_query(const float* x) override {
        q = x;
    }

    void set_query_storage(int idx) override {
        q = &storage[idx * d];
    }
};
