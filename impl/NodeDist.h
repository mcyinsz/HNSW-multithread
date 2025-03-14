#include <iostream>
#include <vector>
#include <algorithm>
#pragma once

// std::queue<NodeDistCloser> to build a maxheap
struct NodeDistCloser {
    float d;
    int id;
    NodeDistCloser(float d, int id) : d(d), id(id) {}
    bool operator<(const NodeDistCloser& obj1) const {
        return d < obj1.d;
    }
};


// std::queue<NodeDistCloser> to build a minheap
struct NodeDistFarther {
    float d;
    int id;
    NodeDistFarther(float d, int id) : d(d), id(id) {}
    bool operator<(const NodeDistFarther& obj1) const {
        return d > obj1.d;
    }
};