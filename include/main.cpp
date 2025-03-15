#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_set>
#include "utils/Index.h"
#include <impl/IndexHNSW.h>
#include <algorithm>

// generate random float number
float random_float(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

// generate random vector
std::vector<float> generate_random_vector(int d, float min, float max) {
    std::vector<float> vec(d);
    for (int i = 0; i < d; ++i) {
        vec[i] = random_float(min, max);
    }
    return vec;
}

// calculate recall
float calculate_recall(const std::vector<int>& hnsw_labels, const std::vector<int>& flat_labels) {
    std::unordered_set<int> flat_set(flat_labels.begin(), flat_labels.end());
    int common = 0;
    for (int label : hnsw_labels) {
        if (flat_set.find(label) != flat_set.end()) {
            common++;
        }
    }
    return static_cast<float>(common) / flat_labels.size();
}

void validate_avx() {
    std::vector<float> data = generate_random_vector(256, -3.0f, 3.0f);
    GenericDistanceComputerIP dc_orig(data, 256);  // 原始版本
    GenericDistanceComputerIP_AVX2 dc_simd(data, 256);  // SIMD版本
    GenericDistanceComputerIP_AVX512 dc_simd512(data, 256);
    
    float q[128];
    std::fill_n(q, 128, 2.0f);
    dc_orig.set_query(q);
    dc_simd.set_query(q);
    dc_simd512.set_query(q);
    
    assert(dc_orig(0) - dc_simd(0) < float(1e-20));  // 验证计算结果一致
    assert(dc_orig(0) - dc_simd512(0) < float(1e-20));
    std::cout << "  validation passed "  << std::endl;
    
}

int main() {

    // validate avx version of distance computer
    validate_avx();

    // parameter settings
    int d = 128;  // vector dimension
    int n = 131072;  // vector number
    int n_query = 1;  // query vector number
    int k = 128;  // the kNN's K
    int efConst = 24;

    // IndexHNSW using metric INNER_PRODUCT
    IndexHNSW index(d, 24, INNER_PRODUCT);
    index.hnsw.efConstruction = efConst;

    // Flat index for reference
    IndexFlat index_flat(d, INNER_PRODUCT);

    // generate random vectors and add them into index
    std::vector<float> vectors;
    for (int i = 0; i < n; ++i) {
        auto vec = generate_random_vector(d, -3.0f, 3.0f);
        vectors.insert(vectors.end(), vec.begin(), vec.end());
    }
    
    std::cout << "  starting adding vectors "  << std::endl;
    // record building time 
    auto start = std::chrono::high_resolution_clock::now();
    index.add(n, vectors);
    std::cout << "  end adding vectors "  << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    // output building time
    std::cout << "Add completed in " << elapsed1.count() << " seconds." << std::endl;

    // index.hnsw.print_neighbor_stats(0);

    // Flat index add
    std::cout << "  starting adding vectors "  << std::endl;
    // record building time 
    start = std::chrono::high_resolution_clock::now();
    index_flat.add(n, vectors);
    std::cout << "  end adding vectors "  << std::endl;
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_flat = end - start;
    // output building time
    std::cout << "Add completed in " << elapsed_flat.count() << " seconds." << std::endl;

    // generate random query vectors
    std::vector<float> query_vectors;
    for (int i = 0; i < n_query; ++i) {
        auto vec = generate_random_vector(d, -3.0f, 3.0f);
        query_vectors.insert(query_vectors.end(), vec.begin(), vec.end());
    }

    // // add queries themselves for benchmark
    // index.add(n_query, query_vectors);
    // index_flat.add(n_query, query_vectors);

    // allocators for query
    std::vector<std::vector<float>> distances;
    std::vector<std::vector<int>> labels;

    std::vector<std::vector<float>> distances_flat;
    std::vector<std::vector<int>> labels_flat;

    // record searching time
    start = std::chrono::high_resolution_clock::now();
    // execute searching
    index.search(n_query, query_vectors, k, distances, labels, 1280);
    

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;

    // output results
    std::cout << "Search completed in " << elapsed2.count() << " seconds." << std::endl;

    // record searching time Flat
    start = std::chrono::high_resolution_clock::now();
    // execute searching
    index.search(n_query, query_vectors, k, distances, labels, 1000);  
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2_flat = end - start;
    index_flat.search(n_query, query_vectors, k, distances_flat, labels_flat);
    std::cout << "Flat Search completed in " << elapsed2_flat.count() << " seconds." << std::endl;

    // output results for first 5 results and recall
    for (int i = 0; i < std::min(5, n_query); ++i) {
        // std::cout << "Query " << i << " results:" << std::endl;
        // for (int j = 0; j < k; ++j) {
        //     std::cout << "  Distance: " << distances[i][j] << ", Label: " << labels[i][j] << std::endl;
        // }
        // for (int j = 0; j < k; ++j) {
        //     std::cout << "  Distance flat: " << distances_flat[i][j] << ", Label flat: " << labels_flat[i][j] << std::endl;   
        // }

        // calculating recall
        float recall = calculate_recall(labels[i], labels_flat[i]);
        std::cout << "Recall for query " << i << ": " << recall << std::endl;
    }

    return 0;
}
