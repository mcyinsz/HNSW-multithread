#include <vector>
#include <utils/constants.h>
#include <utils/DistanceComputer.h>
#include <cassert>
#include <utils/ResultHandler.h>
#include <stdexcept> 
#include <omp.h>
// #include <DistanceComputer.cpp>
#pragma once

class Index {
public:
    int d;  // vector dimension
    int ntotal;  // total vector number
    bool verbose;
    bool is_trained;
    // int metric_type;
    MetricType metric_type;

    Index(int d = 0, MetricType metric = INNER_PRODUCT) 
        : d(d), ntotal(0), verbose(false), is_trained(true), metric_type(metric) {}

    virtual void add(int n, const std::vector<float>& x) = 0;
    virtual void search(int n, const std::vector<float>& x, int k, 
                        std::vector<std::vector<float>>& distances, 
                        std::vector<std::vector<int>>& labels,
                        int Param_efSearch) = 0;
    virtual ~Index() = default;
};

class IndexFlat : public Index {
public:
    std::vector<float> vectors;  // vector data stored in vector structure

    IndexFlat(int d = 0, MetricType metric = INNER_PRODUCT) : Index(d, metric) {}

    void add(int n, const std::vector<float>& x) override {
        assert(x.size() == n * d);
        vectors.insert(vectors.end(), x.begin(), x.end()); // insert all the elements into vectors
        ntotal += n;
    }

    DistanceComputer* get_distance_computer() const {
        if (metric_type == INNER_PRODUCT) {
            return new GenericDistanceComputerIP_AVX512(vectors, d);
        } else if (metric_type == L2_DISTANCE) {
            return new GenericDistanceComputerL2_AVX512(vectors, d);
        } else if (metric_type == COSINE_SIMILARITY) {
            return new GenericDistanceComputerCosine_AVX512(vectors, d);
        }
        return nullptr;
    }

    // get_distance_computer with parsed parameter
    DistanceComputer* get_distance_computer(MetricType parsed_metric_type) const {
        if (parsed_metric_type == INNER_PRODUCT) {
            return new GenericDistanceComputerIP_AVX512(vectors, d);
        } else if (parsed_metric_type == L2_DISTANCE) {
            return new GenericDistanceComputerL2_AVX512(vectors, d);
        } else if (parsed_metric_type == COSINE_SIMILARITY) {
            return new GenericDistanceComputerCosine_AVX512(vectors, d);
        }
        return nullptr;
    }

    void search(int n, const std::vector<float>& x, int k, 
                std::vector<std::vector<float>>& distances, 
                std::vector<std::vector<int>>& labels,
                int Param_efSearch = 0) override {


        std::vector<HeapResultHandler> res_list(n, HeapResultHandler(k));

        #pragma omp parallel for schedule(static) // parallel searching
        for (int i = 0; i < n; ++i) {

            DistanceComputer* qdis = get_distance_computer();

            if (qdis == nullptr) {
                throw std::runtime_error("Distance computer is not initialized");
            }
            
            const float* query_ptr = x.data() + i * d; // set the ith query vector to be the query of the distance computer
            qdis->set_query(query_ptr);

            for (int index = 0; index < ntotal; ++index) {
                float distance = (*qdis)(index);
                res_list[i].add_result(distance, index);
            }

            delete qdis;
        }

        for (int i = 0; i < n; ++i) {
            auto [distance, index] = res_list[i].end();
            if (metric_type == INNER_PRODUCT) {
                std::transform(distance.begin(), distance.end(), distance.begin(), [](float x) { return -x; }); // the instant function
            }
            distances.push_back(distance);
            labels.push_back(index);
        }

        
    }
};
