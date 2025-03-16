#include <include/impl/IndexHNSW.h>
#include <include/utils/DistanceComputer.h>
#include <include/utils/Index.h>
#include <vector>
#include <memory>
#include <include/utils/ResultHandler.h>

#include <omp.h>

using storage_idx_t = HNSW::storage_idx_t ;

DistanceComputer* storage_distance_computer(const IndexFlat* storage) {
    return storage->get_distance_computer();
}

void hnsw_add_vertices(
        IndexHNSW& index_hnsw,
        int n0,
        int n,
        const std::vector<float>& x,
        bool preset_levels = false) {

        int d = index_hnsw.d;

        
        HNSW& hnsw = index_hnsw.hnsw;
        size_t ntotal = n0 + n;

        if (n == 0) {
            return;
        }

        int max_level = hnsw.prepare_level_tab(n, preset_levels);

        std::vector<omp_lock_t> locks(ntotal);
        for (int i = 0; i < ntotal; i++)
            omp_init_lock(&locks[i]);


        // add vectors from highest to lowest level
        std::vector<int> hist;
        std::vector<int> order(n);

        { // make buckets with vectors of the same level

            // build histogram
            for (int i = 0; i < n; i++) {
                storage_idx_t pt_id = i + n0;
                int pt_level = hnsw.levels[pt_id] - 1;
                while (pt_level >= hist.size())
                    hist.push_back(0);
                hist[pt_level]++;
            }

            // accumulate
            std::vector<int> offsets(hist.size() + 1, 0);
            for (int i = 0; i < hist.size() - 1; i++) {
                offsets[i + 1] = offsets[i] + hist[i];
            }

            // bucket sort
            for (int i = 0; i < n; i++) {
                storage_idx_t pt_id = i + n0;
                int pt_level = hnsw.levels[pt_id] - 1;
                order[offsets[pt_level]++] = pt_id;
            }
        }
        


        { // perform add
            RandomGenerator rng2(789);

            int i1 = n;

            for (int pt_level = hist.size() - 1;
                pt_level >= !index_hnsw.init_level0;
                pt_level--) {


                int i0 = i1 - hist[pt_level];

                // random permutation to get rid of dataset order bias
                for (int j = i0; j < i1; j++)
                    std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);


    #pragma omp parallel if (i1 > i0 + 100)
                {
                    VisitedTable vt(ntotal);

                    std::unique_ptr<DistanceComputer> dis(
                            storage_distance_computer(index_hnsw.storage));

                    size_t counter = 0;


                    // here we should do schedule(dynamic) 
    #pragma omp for schedule(dynamic)
                    for (int i = i0; i < i1; i++) {

                        storage_idx_t pt_id = order[i];

                        const float* query_ptr = x.data() + (pt_id - n0) * d;
                        dis->set_query(query_ptr);

                        // cannot break

                        hnsw.add_with_locks(
                                *dis,
                                pt_level,
                                pt_id,
                                locks,
                                vt,
                                index_hnsw.keep_max_size_level0 && (pt_level == 0));

                        counter++;
                    }
                }

                i1 = i0;
            }
            if (index_hnsw.init_level0) {
                assert(i1 == 0);
            } else {
                assert((i1 - hist[0]) == 0);
            }
        }

        for (int i = 0; i < ntotal; i++) {
            omp_destroy_lock(&locks[i]);
        }

}

IndexHNSW::IndexHNSW(int d, int M, int metric)
        : Index(d, metric), hnsw(M), storage(new IndexFlat(d, metric)) {}


void IndexHNSW::add(int n, const std::vector<float>& x) {
    int n0 = ntotal;
    storage->add(n, x);

    ntotal = storage->ntotal;

    hnsw_add_vertices(*this, n0, n, x, hnsw.levels.size() == ntotal);
}

void hnsw_search(
    const IndexHNSW* index,
    int n,
    const std::vector<float>& x,
    std::vector<HeapResultHandler>& bres,
    int Param_efSearch) {

    const HNSW& hnsw = index->hnsw;

    int efSearch = Param_efSearch;

    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(index->storage));
    
    for (int query_number = 0; query_number < n; query_number ++){
        VisitedTable vt(index->ntotal);
        HeapResultHandler& res = bres[query_number]; // citation! not duplication!

        const float* query_ptr = x.data() + query_number * (index->d);
        dis->set_query(query_ptr);

        hnsw.search(*dis, res, vt, Param_efSearch);
    }
}

void IndexHNSW::search(
    int n, 
    const std::vector<float>& x, 
    int k, 
    std::vector<std::vector<float>>& distances, 
    std::vector<std::vector<int>>& labels,
    int Param_efSearch) {
    
    assert(k > 0);

    std::vector<HeapResultHandler> res_list; // 定义一个 vector
    res_list.reserve(n); // 预分配内存，避免多次重新分配
    
    for (int i = 0; i < n; ++i) {
        res_list.emplace_back(k); // 使用 emplace_back 直接构造对象
    }

    hnsw_search(this, n, x, res_list, Param_efSearch);

    for (int i = 0; i < n; ++i) {
        auto [distance, index] = res_list[i].end();
        if (metric_type == INNER_PRODUCT) {
            std::transform(distance.begin(), distance.end(), distance.begin(), [](float x) { return -x; }); // the instant function
        }
        distances.push_back(distance);
        labels.push_back(index);
    }


}