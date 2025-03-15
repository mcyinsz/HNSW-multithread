#include <vector>
#include <bits/types.h>
#include <cstring>
#include <stdexcept>

// the Visited Table for graph walking
struct VisitedTable {
    typedef __uint8_t uint8_t;
    std::vector<uint8_t> visited;
    uint8_t visno;

    explicit VisitedTable(int size) : visited(size), visno(1) {

        if (size < 0) {
            throw std::invalid_argument("Size cannot be negative");
        }

    }

    /// set flag #no to true
    void set(int no) {
        visited[no] = visno;
    }

    /// get flag #no
    bool get(int no) const {
        return visited[no] == visno;
    }

    /// reset all flags to false
    void advance() {
        visno++;
        if (visno == 250) {
            // 250 rather than 255 because sometimes we use visno and visno+1
            memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
            visno = 1;
        }
    }
};