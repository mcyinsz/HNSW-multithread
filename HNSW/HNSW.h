# pragma once

#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <queue>
#include <unordered_set>
#include <random>
#include <cmath>
#include <limits>
#include <cassert>

class HNSW {
public:
    using storage_idx_t = int32_t;  // 存储索引类型
    using Node = std::pair<float, storage_idx_t>;  // 节点类型（距离，索引）

    // 构造函数
    HNSW(int M = 16, int efConstruction = 200, int maxNeighbors = 200);

    // 添加向量
    void add_with_locks(const std::vector<float>& point);

    // 搜索最近邻
    std::vector<storage_idx_t> search(const std::vector<float>& query, int k, int efSearch = 100);

private:
    // 随机生成层级
    int randomLevel();

    // 搜索层级
    void searchLevel(const std::vector<float>& query, std::priority_queue<Node>& candidates, int ef, int level);

    // 数据存储
    std::vector<std::vector<float>> data;  // 存储所有向量
    std::vector<int> levels;               // 每个向量的层级
    std::vector<std::vector<storage_idx_t>> neighbors;  // 每个向量的邻居

    // HNSW 参数
    int M;                // 每层的最大邻居数
    int efConstruction;   // 构建时的扩展因子
    int maxNeighbors;     // 最大邻居数
    int maxLevel;         // 最大层级
    storage_idx_t entryPoint;  // 入口点

    // 随机数生成器
    std::mt19937 rng;
    std::uniform_real_distribution<float> randUniform;
};

#endif // HNSW_H
