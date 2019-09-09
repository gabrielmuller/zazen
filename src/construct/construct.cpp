#pragma once

#include "../render/block.cpp"
#include "../render/voxel.cpp"
#include "indexedleaf.cpp"
#include <iostream>

struct Node {
    union {
        Voxel voxel;
        Leaf leaf;
    };

    bool is_leaf;
    bool is_valid;
    explicit Node() : is_leaf(false), is_valid(false) {}
    Node(const Leaf& leaf) : leaf(leaf), is_leaf(true), is_valid(true) {}
    Node(const Voxel& voxel) : voxel(voxel), is_leaf(false), is_valid(true) {}
};

class NodeQueue {
    Node data[8];
    uint8_t size = 0;

  public:
    void push(const Node& node) {
        data[size] = node;
        size++;
    }

    bool full() const {
        return size == 8;
    }

    void clear() {
        size = 0;
    }

    Node& operator[](uint8_t index) {
        return data[index];
    }

    std::string s() {
        std::string ret = "";
        for (auto i = 0; i < 8; i++) {
            if (i >= size)
                ret += " ";
            else if (data[i].is_leaf)
                ret += "L";
            else if (data[i].is_valid)
                ret += "V";
            else
                ret += "_";
        }
        return ret;
    }
};

class Builder {
    NodeQueue* queues;
    Block* const block;
    const unsigned int queue_count;
    unsigned int stream_index = 0;

    void s() {
        std::cout << "\nIndex " << stream_index << "\n";
        for (unsigned int i = 0; i < queue_count; i++) 
            std::cout << "|" << queues[i].s() << "|\n";
        return;
    }

    void collapse() {
        unsigned int i = 0;
        while (queues[i].full()) {
            /* Collapse full queue i to a voxel in queue (i + 1). */

            NodeQueue& child_queue = queues[i];

            /* If parent is non-leaf, this voxel will be inserted. */
            Voxel parent;
            parent.valid = 0;
            parent.leaf = 0;
            parent.child = block->size();

            /* True at the end when all 8 nodes are the same leaf.
             * Therefore parent will also be that leaf.
             */
            bool parent_is_leaf = true;

            for (unsigned int j = 0; j < 8; j++) {
                /* For each node j in this queue: */

                /* Alias for current node */
                const Node& node = child_queue[j];

                if (node.is_valid) {
                    /* Valid node, mark parent mask. */
                    parent.valid ^= 1 << j;
                } else {
                    /* Invalid node, go to next iteration. */
                    parent_is_leaf = false;
                    continue;
                }

                if (node.is_leaf) {
                    /* Leaf node, mark parent mask. */
                    parent.leaf ^= 1 << j;

                    if (parent_is_leaf  && child_queue[0].leaf != node.leaf) {
                        /* Node is a different leaf. */
                        parent_is_leaf = false;
                    }
                } else {
                    /* Non-leaf node, thus parent is not leaf. */
                    parent_is_leaf = false;
                }
            }
            i++; /* Now working with the (i + 1) queue. */
            NodeQueue& parent_queue = queues[i];


            Node to_push;
            if (parent_is_leaf) {
                /* Parent is leaf, enqueue leaf. */
                to_push = Node(child_queue[0].leaf);
            } else if (!parent.valid) {
                /* No valid children, thus parent is invalid. */
                to_push = Node();
            } else {
                /* Parent is valid and non-leaf, write children and 
                 * enqueue parent voxel. 
                 */
                for (unsigned int j = 0; j < 8; j++) {
                    Node& node = child_queue[j];
                    if (node.is_leaf)
                        new (block->slot()) Leaf(node.leaf);
                    else if (node.is_valid)
                        new (block->slot()) Voxel(node.voxel);
                }
                to_push = Node(parent);
            }

            if (i < queue_count) {
                parent_queue.push(to_push);
            }

            /* All children have been collapsed, therefore clear. */
            child_queue.clear();
        }
    }

  public:
    Builder(Block* block, const unsigned int queue_count) 
            : block(block), queue_count(queue_count) {
        queues = new NodeQueue[queue_count];
    }

    ~Builder() {
        delete[] queues;
    }

    void add_leaf(const IndexedLeaf& ileaf) {
        while (stream_index + 1 < ileaf.index) {
            /* Push invalid nodes so the index catches up with the stream. */
            queues[0].push(Node());
            stream_index++;
            collapse();
        } 

        queues[0].push(ileaf.leaf.valid() ? Node(ileaf.leaf) : Node());
        stream_index++;
        collapse();
    }
};
