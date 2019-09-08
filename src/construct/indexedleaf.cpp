#pragma once
#include "../render/leaf.cpp"

struct IndexedLeaf {
    const Leaf leaf;
    const unsigned int index;
    IndexedLeaf(const Leaf leaf, const unsigned int index) : leaf(leaf), index(index) {}
};

