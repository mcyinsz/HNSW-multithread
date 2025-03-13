#include <vector>
#include <cmath>
#include "distancecomputer.h"

class GenericDistanceComputerIP : public DistanceComputer {
    int d;  // 向量维度
    const std::vector<float>& storage;  // 连续内存存储的向量数据
    std::vector<float> q;  // 查询向量

public:
    GenericDistanceComputerIP(const std::vector<float>& storage, int d) 
        : storage(storage), d(d) {}

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

    void set_query(const std::vector<float>& x) override {
        q = x;
    }

    void set_query_storage(int idx) override {
        q.assign(storage.begin() + idx * d, storage.begin() + (idx + 1) * d);
    }
};

class GenericDistanceComputerL2 : public DistanceComputer {
    int d;  // 向量维度
    const std::vector<float>& storage;  // 连续内存存储的向量数据
    std::vector<float> q;  // 查询向量

public:
    GenericDistanceComputerL2(const std::vector<float>& storage, int d) 
        : storage(storage), d(d) {}

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

    void set_query(const std::vector<float>& x) override {
        q = x;
    }

    void set_query_storage(int idx) override {
        q.assign(storage.begin() + idx * d, storage.begin() + (idx + 1) * d);
    }
};
