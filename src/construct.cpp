#pragma once

#include "model.cpp"
#include "block.cpp"
#include "leaf.cpp"

struct Node;

struct InternalNode {
    Node* children[8];
    explicit InternalNode() {}
};

struct Node {
    enum {LEAF, INTERNAL, INVALID} tag;
    explicit Node() : tag(INVALID) {}
    Node(Leaf leaf) : tag(LEAF), leaf(leaf) {}
    Node(InternalNode node) : tag(INTERNAL), node(node) {}
    union {
        InternalNode node;
        Leaf leaf;
    };
};

Node* traverse(int size, int x, int y, int z, const Model& model) {
    printf("Traverse size %d (%d, %d, %d)\n", size, x, y, z);
    if (x >= model.width || y >= model.height || z >= model.depth) {
        // invalid node
        return new Node();
    }

    if (size == 1) {
        return new Node(model.get(x, y, z));
    }

    InternalNode in;

    for (int i = 0; i < 8; i++) {
        int xi = 4 & i ? x : x + size / 2;
        int yi = 4 & i ? y : y + size / 2;
        int zi = 4 & i ? z : z + size / 2;
        in.children[i] = traverse(size / 2, xi, yi, zi, model);
    }

    for (int i = 1; i < 8; i++) {
        if (in.children[0]->tag != in.children[i]->tag ||
            in.children[i]->tag == Node::INTERNAL) {
            // children of different nature or internal
            return new Node(in);
        }
    }

    // at this point, the children are either all leaves or all invalid.
    // if all invalid, returning first child works.
    // if all leaves, only return first child if leaves are all equal.

    bool all_equal = true;
    for (int i = 1; i < 8; i++) {
        if (in.children[0]->leaf != in.children[i]->leaf) {
            all_equal = false;
            break;
        }
    }

    if (all_equal) {
        for (int i = 1; i < 8; i++) {
            delete in.children[i];
        }
        return in.children[0];
    }

    // different children, return internal node
    return new Node(in);
}

        
Block* construct(const Model& model) {
    printf("Construct model (%d, %d, %d)\n", model.width, model.height, model.depth);
    traverse(512, 0, 0, 0, model);
    return nullptr;
}
