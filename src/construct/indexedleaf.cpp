#pragma once
#include "../render/leaf.cpp"

struct IndexedLeaf {
    const Leaf leaf;
    const unsigned long long int index;
    IndexedLeaf(const Leaf leaf, const unsigned long long int index) 
            : leaf(leaf), index(index) {}
};

