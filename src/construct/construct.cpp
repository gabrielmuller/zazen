#pragma once

#include <iostream>
#include "model.cpp"
#include "../render/block.cpp"
#include "../render/leaf.cpp"
#include "../render/voxel.cpp"

struct Node;

enum Tag : uint8_t {UNDEFINED, LEAF, INTERNAL, INVALID};

struct InternalNode {
    // tags are stored in parent to pack properly inside 8 byte word
    Tag tags[8];
    Node* children[8];

    explicit InternalNode() {
        for (int i = 0; i < 8; i++) {
            children[i] = nullptr;
            tags[i] = UNDEFINED;
        }
    }
};

struct Node {
    union {
        InternalNode* in;
        Leaf* leaf;
    };

    explicit Node() {}
    Node(Leaf* leaf) : leaf(leaf) {}
    Node(InternalNode* in) : in(in) {}
};

struct TaggedNode {
    Node* node;
    const Tag tag;

    TaggedNode(Node* node, Tag tag) : node(node), tag(tag) {}
};

int count;

TaggedNode create_tree(int size, int x, int y, int z, const Model& model) {
    if (x >= model.width || y >= model.height || z >= model.depth) {
        return TaggedNode(nullptr, INVALID);
    }

    if (size == 1) {
        Leaf leaf = model.get(x, y, z);
        if (leaf.a > 0x80) {
            return TaggedNode(new Node(new Leaf(model.get(x, y, z))), LEAF);
        }
        return TaggedNode(nullptr, INVALID);
    }

    InternalNode* in = new InternalNode();

    for (int i = 0; i < 8; i++) {
        int xi = 4 & i ? x : x + size / 2;
        int yi = 2 & i ? y : y + size / 2;
        int zi = 1 & i ? z : z + size / 2;

        TaggedNode tree = create_tree(size / 2, xi, yi, zi, model);
        in->children[i] = tree.node;
        in->tags[i] = tree.tag;
    }

    for (int i = 1; i < 8; i++) {
        if (in->tags[0] != in->tags[i] ||
            in->tags[i] == INTERNAL) {
            // children of different nature or internal
            count++;
            return TaggedNode(new Node(in), INTERNAL);
        }
    }

    // at this point, the children are either all leaves or all invalid.
    // if all invalid, returning first child works.
    // if all leaves, only return first child if leaves are all equal.

    bool all_equal = true;
    Tag tag = in->tags[0];

    if (tag == LEAF) {
        for (int i = 1; i < 8; i++) {
            if (*(in->children[0]->leaf) != *(in->children[i]->leaf)) {
                all_equal = false;
                break;
            }
        }
    }

    if (all_equal) {
        Node* child = in->children[0];
        if (tag == LEAF) {
            for (int i = 1; i < 8; i++) {
                delete in->children[i]->leaf;
                delete in->children[i];
            }
        }
        delete in;
        return TaggedNode(child, tag);
    }

    // different children, return internal node
    count++;
    return TaggedNode(new Node(in), INTERNAL);
}

void flatten_tree(InternalNode* in,
                  Block* block,
                  Voxel* parent) {
    parent->child = block->size();
    parent->leaf  = 0x00;
    parent->valid = 0xff;

    Voxel* children[8]; // voxel children
    unsigned int i_children[8]; // voxel child #j is #i_children[j] child
    size_t n_children = 0; // number of voxel children

    // FIRST LOOP: create children
    for (int i = 0; i < 8; i++) {
        switch (in->tags[i]) {
            case LEAF:
                new (block->slot()) Leaf(*(in->children[i]->leaf));
                parent->leaf |= 1 << i;
                break;
            case INTERNAL:
                //TODO
                children[n_children] = new (block->slot()) Voxel();
                i_children[n_children] = i;
                n_children++;
                break;
            case INVALID:
                parent->valid ^= 1 << i;
                break;
        }
    }

    // SECOND LOOP: tell children to create their own children
    for (int i = 0; i < n_children; i++) {
        flatten_tree(in->children[i_children[i]]->in, block, children[i]);
    }
}
        
void construct(Model* model, Block* block, Voxel* root_voxel) {
    count = 0;
    Node* root_node = create_tree(512, 0, 0, 0, *model).node;
    delete model;
    std::cout << "Nodes:" << count << "\n";
    flatten_tree(root_node->in, block, root_voxel); 
    std::cout << "Block created. (" << block->size() << "/"
              << block->capacity() << ")\n";
}

inline void construct(Model* model, Block* block) {
    return construct(model, block, new (block->slot()) Voxel());
}
