#pragma once

#include "../render/block.cpp"
#include "../render/voxel.cpp"
#include "indexedleaf.cpp"

struct Node {
    union {
        Voxel voxel;
        Leaf leaf;
    };

    bool is_leaf;
    bool is_valid;
    explicit Node() : is_leaf(false), is_valid(false) {}
    Node(Leaf& leaf) : leaf(leaf), is_leaf(true), is_valid(true) {}
    Node(Voxel& voxel) : voxel(voxel), is_leaf(false), is_valid(true) {}
};

class NodeQueue {
    Node data[8];
    uint8_t size = 0;

  public:
    void push(Node&& node) {
        data[size] = node;
        size++;
    }

    bool full() const {
        return size == 8;
    }

    void collapse() {
        size = 0;
    }

    Node& operator[](uint8_t index) {
        return data[index];
    }
};

class Builder {
    NodeQueue* queues;
    Block* const block;
    const unsigned int queue_count;
    unsigned int stream_index = 0;
    bool _is_done = false;

  public:
    Builder(Block* block, const unsigned int queue_count) 
            : block(block), queue_count(queue_count) {
        queues = new NodeQueue[queue_count];
    }

    ~Builder() {
        delete[] queues;
    }

    void add_leaf(IndexedLeaf ileaf) {
        int i = queue_count - 1;
        Leaf leaf = ileaf.leaf;
        unsigned int index = ileaf.index;

collapse:
        while (queues[i].full()) {
            /* Collapse full queue on top to a voxel in the queue below. */
            NodeQueue& queue = queues[i];
            Voxel parent;
            parent.valid = 0;
            parent.leaf = 0;
            parent.child = block->size();
            bool parent_is_leaf = true;
            for (auto j = 0; j < 8; j++) {
                Node& node = queue[j];

                if (node.is_valid) {
                    parent.valid ^= 1 << j;
                } else {
                    parent_is_leaf = false;
                    break;
                }

                if (node.is_leaf) {
                    parent.leaf ^= 1 << j;
                    new (block->slot()) Leaf(node.leaf);
                    if (parent_is_leaf  && queue[0].leaf != node.leaf) {
                        parent_is_leaf = false;
                    }
                } else {
                    parent_is_leaf = false;
                    new (block->slot()) Voxel(node.voxel);
                }
            }
            i--;
            if (i < 0) {
                new (block->slot()) Voxel(parent);
                _is_done = true;
                return;
            }

            if (parent_is_leaf) {
                queue.push(Node(queue[0].leaf));
            } else {
                queue.push(Node(parent));
            }
        }

        if (stream_index + 1 < index) {
            queues[0].push(Node());
            stream_index++;
            goto collapse;
        }

        queues[0].push(leaf);
        stream_index++;
    }

    inline bool is_done() { return _is_done; }
};
