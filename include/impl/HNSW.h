#ifndef HNSW_H  // 防止重复包含
#define HNSW_H

#include <string>  // 包含必要的头文件
#include <vector>
#include <utils/RandomGenerator.h>
#include <omp.h>
#include <utils/DistanceComputer.h>
#include <utils/VisitedTable.h>
#include <impl/NodeDist.h>
#include <queue>
#include <utils/ResultHandler.h>
#include <stdexcept> 

class HNSW {
public:
    using storage_idx_t = int32_t;
    using idx_t = int64_t;
    // construct function
    explicit HNSW(int M = 32);

    // ============================================
    // basic functions
    // ============================================
    void reset(); // reset the inner states

    // ============================================
    // for build HNSW structure
    // ============================================

    // set default degree limits
    void set_default_probas(int M, float levelMult);

    // set the number of neighbors for certain level
    void set_nb_neighbors(int level_no, int n);

    // number of neighbors for this level
    int nb_neighbors(int layer_no) const;

    /// cumumlative nb up to (and excluding) this level
    int cum_nb_neighbors(int layer_no) const;

    // range of entries in the neighbors table of nodes number at layer_no
    void neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const;

    // return a random level
    int random_level();

    // prepare level tables for new add nodes
    int prepare_level_tab(
        size_t n,
        bool preset_levels = false
    );

    // add links for a given point in given level
    // the input level is commonly set 0
    void add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest, // previous nearest (probably from the greedy search)
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        bool keep_max_size_level0 = false
    );

    // add point pt_id on all levels <= pt_level and build the link
    void add_with_locks(
        DistanceComputer&ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt,
        bool keep_max_size_level0 = false
    );

    void print_neighbor_stats(int level) const;

    // ============================================
    // for search on HNSW structure
    // ============================================

    void search(
        DistanceComputer& qdis,
        HeapResultHandler& res, // different from FAISS, the result handler would only store the small distance
        VisitedTable&  vt,
        int Param_efSearch = 16 // different from FAISS, no HNSW Search Parameters
    ) const;

    // class method
    static void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size,
        bool keep_max_size_level0 = false);

    // functional functions
    friend void greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest);

    friend int count_below(const std::vector<float>& previous_vectors, float d0);

    friend void search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        HeapResultHandler& res,
        std::vector<NodeDistFarther>& vec_candidates, // attention, the data structure is changed to std::vector
        VisitedTable& vt,
        int level,
        int nres_in,
        int Param_efSearch);
    
    

    // data structure

// private:
    // member varients
    
    // ============================================
    // for build HNSW structure
    // ============================================
    
    // assignment probability to each layer (sum = 1)
    std::vector<double> assign_probas;
    
    // number of neighbors stored per layer (cumulative)
    // should not be changed since the first add operation
    std::vector<int> cum_num_neighbor_per_level;

    // (level + 1) of each vector
    std::vector<int> levels;

    // the [offsets[i], offsets[i + 1]) is the `neighors` indices
    // storage range for node with index i
    std::vector<size_t> offsets;
    std::vector<storage_idx_t> neighbors;

    // random generator
    RandomGenerator rng;

    // Construction searching width
    int efConstruction = 40;

    // ============================================
    // for search HNSW structure
    // ============================================

    // entry point for add and search
    storage_idx_t entry_point = -1;

    // max level
    int max_level = -1;

    // beam search width
    int efSearch = 16;

};

#endif  // MYCLASS_H