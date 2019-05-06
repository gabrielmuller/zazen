#pragma once

#include "model.cpp"
#include "block.cpp"
#include "leaf.cpp"

struct InternalNode {
    InternalNode* children[8];
    explicit InternalNode() {}
    ~InternalNode() {
        for (int i = 0; i < 8; i++) {
            delete children[i];
        }
    }
};

struct Node {
    enum {INTERNAL, LEAF} tag;
    union {
        InternalNode node;
        Leaf leaf;
    };
};

Block* construct(const Model& model) {
    return nullptr;
}
