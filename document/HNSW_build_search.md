# HNSW build and search detailed steps

## HNSW build

### Call chain

1. `IndexHNSW::add`
    * add vectors into inner flat index
    * call `hnsw_add_vertives`, with before/after storage size
2. `hnsw_add_vertices`
    * prepare level table for newly added vectors, call `HNSW::prepare_level_tab`
        * prepare `hnsw.offset` array for vectors (offset[i+1] - offset[i] is the neighbor list for vector i)
        * prepare `levels` array
        * pre-set neighbor positions to `-1`
    * initialize locks for all the vectors (previous/to-be-added)
    * do the bucket sort for all the to-be-added vectors
    * swap the nodes in the same level
    * initialize `VisitedTable` for graph walking
    * initialize Distance computer with `storage_distance_computer`
    * call `HNSW::add_with_locks` for each new node
    * destroy locks
3. `HNSW::add_with_locks`
    * set lock for current point (for neighbor list)
    * `greedy_update_nearest` for level > 0
    * `add_links_starting_from` for level = 0
    * unset lock
4. `greedy_update_nearest`
    * greedy pick the nearest neighbor, then walk to the new node (with distance calculation)
5. `HNSW::add_links_starting_from`
    * call `search_neighbors_to_add`
    * call `shrink_neighbor_list` to shrink the efConstruction-size neighbor list to M-size list, heuristic shrink approach to satisfy degree limit (with distance calculation)
    * call `add_link` to add out/in edges (with distance calculation)
6. `add_link`
    * if there is enough room, just fill up the empty position
    * else, should re-calculate the distances between node and its neighbors (with distance calculation), then call `shrink_neighbor_list` (with distance calculation)

### illustration

![alt text](/pics/HNSW_build_figure.png)

## HNSW search

### Call chain

1. `IndexHNSW::search`
    * initialize result handler for each query vectors
    * call `hnsw_search`
    * output each result handler's results
2. `hnsw_search`
    * for each query vector, initialize its distance computer, visitedtable
    * call `HNSW::search`
3. `HNSW::search`
    * from `entry_point` and `d_nearest`, greedy update nearest on level > 0 layers (with distance calculation)
    * call `search_from_candidates`
4. `search_from_candidates`
    * beam search (with distance calculation)

    
### illustration

![alt text](/pics/HNSW_search_figure.png)
