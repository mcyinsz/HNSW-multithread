#include <vector>
#include <utils/constants.h>
#include <utils/DistanceComputer.h>
#include <cassert>
#include <utils/ResultHandler.h>
#include <utils/DistanceComputer.cpp>

class Index {
public:
    int d;  // 向量维度
    int ntotal;  // 总向量数量
    bool verbose;
    bool is_trained;
    int metric_type;

    Index(int d = 0, int metric = INNER_PRODUCT) 
        : d(d), ntotal(0), verbose(false), is_trained(true), metric_type(metric) {}

    virtual void add(int n, const std::vector<float>& x) = 0;
    virtual void search(int n, const std::vector<float>& x, int k, 
                        std::vector<std::vector<float>>& distances, 
                        std::vector<std::vector<int>>& labels) = 0;
    virtual ~Index() = default;
};

class IndexFlat : public Index {
public:
    std::vector<float> vectors;  // 连续内存存储的向量数据

    IndexFlat(int d = 0, int metric = INNER_PRODUCT) : Index(d, metric) {}

    void add(int n, const std::vector<float>& x) override {
        assert(x.size() == n * d);
        vectors.insert(vectors.end(), x.begin(), x.end()); // insert all the elements into vectors
        ntotal += n;
    }

    DistanceComputer* get_distance_computer() {
        if (metric_type == INNER_PRODUCT) {
            return new GenericDistanceComputerIP(vectors, d);
        } else if (metric_type == L2_DISTANCE) {
            return new GenericDistanceComputerL2(vectors, d);
        }
        return nullptr;
    }

    void search(int n, const std::vector<float>& x, int k, 
                std::vector<std::vector<float>>& distances, 
                std::vector<std::vector<int>>& labels) override {
        std::vector<HeapResultHandler> res_list(n, HeapResultHandler(k));
        DistanceComputer* qdis = get_distance_computer();

        for (int i = 0; i < n; ++i) {
            
            const float* query_ptr = x.data() + i * d; // set the ith query vector to be the query of the distance computer
            qdis->set_query(query_ptr);

            for (int index = 0; index < ntotal; ++index) {
                float distance = (*qdis)(index);
                res_list[i].add_result(distance, index);
            }
        }

        for (int i = 0; i < n; ++i) {
            auto [distance, index] = res_list[i].end();
            if (metric_type == INNER_PRODUCT) {
                std::transform(distance.begin(), distance.end(), distance.begin(), [](float x) { return -x; }); // the instant function
            }
            distances.push_back(distance);
            labels.push_back(index);
        }

        delete qdis;
    }
};
