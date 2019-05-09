#pragma once

#include <iostream>
#include "model.cpp"
#include "../render/block.cpp"
#include "../render/leaf.cpp"

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

int count = 0;
Node* create_tree(int size, int x, int y, int z, const Model& model) {
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
        in.children[i] = create_tree(size / 2, xi, yi, zi, model);
    }

    for (int i = 1; i < 8; i++) {
        if (in.children[0]->tag != in.children[i]->tag ||
            in.children[i]->tag == Node::INTERNAL) {
            // children of different nature or internal
            count++;
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

unsigned int flatten_tree(InternalNode* in, Block* block, unsigned int offset) {
    for (int i = 0; i < 8; i++) {
        const Node& child = *(in->children[i]);
        switch child.tag {
            case Node::LEAF:
                new (block->slot()) Leaf(child.leaf);
                return ++offset;
            case Node::INTERNAL:
                //TODO
                break;
        }
    }
    return 0;
}
        
Block* construct(Model& model) {
    Node* root = create_tree(512, 0, 0, 0, model);
    delete model;
    printf("count %d\n", count);
    block = new Block(count);
    return nullptr;
}
