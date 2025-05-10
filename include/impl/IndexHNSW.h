#pragma once

#include <vector>
#include <utils/Index.h>
#include <impl/HNSW.h>
#include <utils/constants.h>

class IndexHNSW: public Index{
    typedef HNSW::storage_idx_t storage_idx_t;
    typedef HNSW::idx_t idx_t;
    public:

        // graph structure
        HNSW hnsw;

        // flat storage
        IndexFlat* storage;

        // init level 0 graph
        bool init_level0 = true;

        // reach the degree limit in graph level 0
        bool keep_max_size_level0 = false;

        // search metric type
        MetricType search_metric = INNER_PRODUCT;

        explicit IndexHNSW(int d = 0, int M = 32, MetricType metric = INNER_PRODUCT);
        
        explicit IndexHNSW(int d, int M, MetricType metric, MetricType search_metric);

        ~IndexHNSW() override;

        void add(int n, const std::vector<float>& x) override; // the type is different from FAISS

        void search(
            int n, const std::vector<float>& x, int k, 
                        std::vector<std::vector<float>>& distances, 
                        std::vector<std::vector<int>>& labels,
                        int Param_efSearch) override;

        // void reset();

};