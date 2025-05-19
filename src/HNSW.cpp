#include <impl/HNSW.h>
#include <impl/NodeDist.h>
#include <utils/ResultHandler.h>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <set>
#include <unordered_set>

using storage_idx_t = HNSW::storage_idx_t;

// ============================================
// basic functions
// ============================================

HNSW::HNSW(int M) : rng(12345) {
    assert(M > 0);
    set_default_probas(M, 1.0 / log(M));
    offsets.reserve(128000 + 1); // preallocate memory
    offsets.push_back(0);
    
    neighbors.reserve(128000 * 2 * M); // preallocate memory for neighbors
}

void HNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

/**************************************************************
 * assissant functions
 **************************************************************/

int count_below(const std::multiset<float>& previous_poped_distance,float d0) {
    return std::distance(previous_poped_distance.begin(), previous_poped_distance.lower_bound(d0));
}

// ============================================
// for build HNSW structure
// ============================================

int HNSW::nb_neighbors(int layer_no) const{
    return cum_num_neighbor_per_level[layer_no + 1] -
    cum_num_neighbor_per_level[layer_no];
}

// change accumulative number of neighbors vector
void HNSW::set_nb_neighbors(int level_no, int n) {
    assert(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);

    for (int i = level_no + 1; i < cum_num_neighbor_per_level.size(); i++){
        cum_num_neighbor_per_level[i] += n - cur_n;
    }
}

int HNSW::cum_nb_neighbors(int layer_no) const{
    return cum_num_neighbor_per_level[layer_no];
}

void HNSW::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

int HNSW::random_level() {
    double f = rng.rand_float();

    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level; // this level is the actually level, while the levels[] would store level + 1
        }
        f -= assign_probas[level];
    }

    return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult) {
    int nn = 0;
    cum_num_neighbor_per_level.push_back(0); // accumulative, so the first element is 0
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assign_probas.push_back(proba);
        nn += level == 0 ? M * 2 : M;
        cum_num_neighbor_per_level.push_back(nn);
    }
}

int HNSW::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    // update levels list
    if (preset_levels) {
        assert(n0 + n == levels.size());
    } else {
        assert(n0 == levels.size());
        for (int i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1); // levels actually stores the level + 1
        }
    }

    int max_level_2 = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level_2)
            max_level_2 = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
    }
    neighbors.resize(offsets.back(), -1);

    return max_level_2; // the newly added nodes' max level
}

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HNSW::shrink_neighbor_list(
    DistanceComputer& qdis,
    std::priority_queue<NodeDistFarther>& input,
    std::vector<NodeDistFarther>& output,
    int max_size,
    bool keep_max_size_level0) {
// This prevents number of neighbors at
// level 0 from being shrunk to less than 2 * M.
// This is essential in making sure
// `faiss::gpu::GpuIndexCagra::copyFrom(IndexHNSWCagra*)` is functional
std::vector<NodeDistFarther> outsiders;

while (input.size() > 0) {
    NodeDistFarther v1 = input.top();
    input.pop();
    float dist_v1_q = v1.d;

    bool good = true;
    for (NodeDistFarther v2 : output) {
        float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

        if (dist_v1_v2 < dist_v1_q) {
            good = false;
            break;
        }
    }

    if (good) {
        output.push_back(v1);
        if (output.size() >= max_size) {
            return;
        }
    } else if (keep_max_size_level0) {
        outsiders.push_back(v1);
    }
}
size_t idx = 0;
while (keep_max_size_level0 && (output.size() < max_size) &&
       (idx < outsiders.size())) {
    output.push_back(outsiders[idx++]);
}
}

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list_single(
    DistanceComputer& qdis,
    std::priority_queue<NodeDistCloser>& resultSet1,
    int max_size,
    bool keep_max_size_level0 = false) {
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSW::shrink_neighbor_list(
            qdis, resultSet, returnlist, max_size, keep_max_size_level0);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}



/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(
    HNSW& hnsw,
    DistanceComputer& qdis,
    storage_idx_t src,
    storage_idx_t dest,
    int level,
    bool keep_max_size_level0 = false) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1)
                break;
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    shrink_neighbor_list_single(qdis, resultSet, end - begin, keep_max_size_level0);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }

}


/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt) {
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        if (currEv.d > results.top().d) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        // the reference version
        for (size_t i = begin; i < end; i++) {
            storage_idx_t nodeId = hnsw.neighbors[i];
            if (nodeId < 0)
                break;
            if (vt.get(nodeId))
                continue;
            vt.set(nodeId);

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId);

            if (results.size() < hnsw.efConstruction ||
                results.top().d > dis) {
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }
        } 
    }

    vt.advance();
}

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {

    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt);

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    shrink_neighbor_list_single(ptdis, link_targets, M, keep_max_size_level0);

    std::vector<storage_idx_t> neighbors_to_add;
    neighbors_to_add.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level0);
        neighbors_to_add.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors_to_add) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level0);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

 void HNSW::add_with_locks(
    DistanceComputer& ptdis,
    int pt_level,
    int pt_id,
    std::vector<omp_lock_t>& locks,
    VisitedTable& vt,
    bool keep_max_size_level0) {
    //  greedy search on upper levels

    storage_idx_t nearest;
    #pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    for (; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
        add_links_starting_from(
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks.data(),
                vt,
                keep_max_size_level0);
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}

/**************************************************************
 * Searching
 **************************************************************/

HNSWStats greedy_update_nearest(
    const HNSW& hnsw,
    DistanceComputer& qdis,
    int level,
    storage_idx_t& nearest,
    float& d_nearest) {

    // initialize probe stats (probe)
    HNSWStats stats;

    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        // the distance number for the current node's neighbors (probe)
        size_t ndis = 0;

        for (size_t i = begin; i < end; i++) {
            storage_idx_t v = hnsw.neighbors[i];
            if (v < 0)
                break;
            ndis += 1;
            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }

        // update stats (probe)
        // one greedy walk step brings one hop number
        stats.ndis += ndis;
        stats.nhops += 1;

        // greedy stop strategy
        if (nearest == prev_nearest) {
            return stats;
        }

    }
}

void HNSW::print_neighbor_stats(int level) const {
    assert(level < cum_num_neighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
        reduction(+ : tot_reciprocal) reduction(+ : n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0)
                    break;
                neighset.insert(neighbors[j]);
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0)
                    break;
                assert(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %zd\n", n_node);
    printf("   neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

HNSWStats HNSW::search(
    DistanceComputer& qdis,
    HeapResultHandler& res,
    VisitedTable& vt,
    int Param_efSearch) const {
    
    // initialize HNSW
    HNSWStats stats;

    // empty HNSW
    if (entry_point == -1) {
        return stats;
    }

    // get kNN's corresponding k
    int k = res.k;

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for (int level = max_level; level >= 1; level--) {

        // the greedy search on each level
        HNSWStats local_stats = 
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);

        // merge level's stats with main stats
        stats.combine(local_stats);

        // probe for once
        // std::cout << "level: " << level << "\t"
        //       << "ndis: " << local_stats.ndis << "\t"
        //       << "nhops: " << local_stats.nhops << "\t"
        //       << "cur ndis: " << stats.ndis << "\t"
        //       << "cur hop: " << stats.nhops << "\t" <<
        //       std::endl;
        }

    // int ef = std::max(params ? params->efSearch : efSearch, k);

    // this is the most common branch
    std::vector<NodeDistFarther> vec_candidates;

    NodeDistFarther nearest_node = NodeDistFarther(d_nearest , nearest);
    vec_candidates.push_back(nearest_node);

    search_from_candidates(
            *this, qdis, res, vec_candidates, vt, stats, 0, 0, Param_efSearch);
    vt.advance();

    // std::cout << "level: " << 0 << "\t"
    //           << "cur ndis: " << stats.ndis << "\t"
    //           << "cur hop: " << stats.nhops << "\t" <<
    //           std::endl;

    return stats;
}

int search_from_candidates(
    const HNSW& hnsw,
    DistanceComputer& qdis,
    HeapResultHandler& res,
    std::vector<NodeDistFarther>& vec_candidates, // attention, the data structure is changed to std::vector
    VisitedTable& vt,
    HNSWStats& stats,
    int level,
    int nres_in,
    int Param_efSearch) {

    // the result handler's result number
    int nres = nres_in;

    // initialize distance calculate time (probe)
    size_t ndis = 0;

    // can be overridden by search params
    int efSearch = Param_efSearch;

    float threshold = res.threshold;

    // init the candidates with minheap structure
    std::priority_queue<NodeDistFarther> candidates;

    // init the previous distances list
    // std::vector<float> previous_poped_distance;
    std::multiset<float> previous_poped_distance;

    // add candidates into result handler
    for (int i = 0; i < vec_candidates.size(); i++) {
        HNSW::idx_t v1 = vec_candidates[i].id;
        float d = vec_candidates[i].d;
        assert(v1 >= 0);

        if (d < threshold) {
            if (res.add_result(d, v1)) {
                threshold = res.threshold;
            }
        }

        candidates.push(vec_candidates[i]); // push candidate into heap
        previous_poped_distance.insert(d);

        vt.set(v1);
    }

    // initialize hop time (probe)
    size_t nstep = 0;

    while (candidates.size() > 0) {
        
        int v0 = candidates.top().id; 
        float d0 = candidates.top().d;

        candidates.pop();
        previous_poped_distance.insert(d0);
        // tricky stopping condition: there are more that ef
        // distances that are processed already that are smaller
        // than d0

        // int n_dis_below = count_below(previous_poped_distance,d0);
        // if (n_dis_below >= efSearch) {
        //     break;
        // }

        // start search in current candidate's neighbors
        size_t begin, end;
        
        hnsw.neighbor_range(v0, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);

            // calculate distance once (probe)
            ndis++;
            float d = qdis(v1);
            
            // try to add result into the result handler
            if (res.add_result(d, v1)) {
                threshold = res.threshold;
                nres += 1;
            } 
            candidates.emplace(d, v1); // add the new candidate into heap
        }

        // add one hop
        nstep++;
        if (ndis > efSearch) { // constraint ndis must lower than efSearch
            break;
        }
    }

    // update probe state
    if (level == 0) {

        // have searched one vector (probe)
        stats.n1++;

        // add one empty candidate queue vector (probe)
        // good for detect unconnected nodes
        if (candidates.size() == 0){
            stats.n2 ++;
        }

        // update calculated distances (probe)
        stats.ndis += ndis;

        // update hop number (probe)
        stats.nhops += nstep;
    }

    return nres;
}

