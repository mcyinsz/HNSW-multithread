# include<queue>
# include <algorithm>

class HeapResultHandler {
    int k;
    std::priority_queue<std::pair<float, int>> max_heap;

public:
    HeapResultHandler(int k) : k(k) {}

    void add_result(float dis, int idx) {
        if (max_heap.size() < k || dis < max_heap.top().first) {
            max_heap.push({dis, idx});
            if (max_heap.size() > k) {
                max_heap.pop();
            }
        }
    }

    std::pair<std::vector<float>, std::vector<int>> end() {
        std::vector<float> distances;
        std::vector<int> indices;
        while (!max_heap.empty()) {
            distances.push_back(max_heap.top().first);
            indices.push_back(max_heap.top().second);
            max_heap.pop();
        }
        std::reverse(distances.begin(), distances.end()); 
        std::reverse(indices.begin(), indices.end());
        return {distances, indices};
    }
};
