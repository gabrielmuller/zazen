#pragma once

#include "../render/voxel.cpp"
#include "indexedleaf.cpp"
#include "writer.cpp"
#include <iostream>
#include <cmath>

struct Node {
    union {
        Voxel voxel;
        Leaf leaf;
    };

    bool is_leaf;
    bool is_valid;
    explicit Node() : is_leaf(false), is_valid(false) {}
    explicit Node(const Leaf& leaf) : leaf(leaf), is_leaf(true), is_valid(true) {}
    explicit Node(const Voxel& voxel) : voxel(voxel), is_leaf(false), is_valid(true) {}
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

    bool empty() const {
        return !size;
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
    const unsigned int queue_count;
    unsigned long long int stream_index = 0;
    BlockWriter& writer;

    void s() {
        std::cout << "\nIndex " << stream_index << "\n";
        for (unsigned int i = 0; i < queue_count; i++) 
            std::cout << "|" << queues[i].s() << "|\n";
        return;
    }

    void collapse(unsigned int i) {
        /* Collapse full queue i to a voxel in queue (i + 1). */

        /* Queue is not full, no need to collapse */
        if (!queues[i].full()) return;

        NodeQueue& child_queue = queues[i];

        /* If parent is non-leaf, this voxel will be inserted. */
        Voxel parent;
        parent.valid = 0;
        parent.leaf = 0;
        parent.child = writer.pos();

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
                    writer << node.leaf;
                else if (node.is_valid)
                    writer << node.voxel;
            }
            to_push = Node(parent);
        }

        if (i < queue_count) push_at(to_push, i); // push to parent
        else { writer << to_push.voxel; std::cout << "ROOT PUSHED\n";} // out of queues, push root voxel

        /* All children have been collapsed, therefore clear. */
        child_queue.clear();
    }

    inline void push_at(const Node& node, const uint8_t index) {
        queues[index].push(node);
        collapse(index);
    }


  public:
    Builder(const unsigned int queue_count, BlockWriter& writer) 
            : queue_count(queue_count), writer(writer) {
        queues = new NodeQueue[queue_count];
    }

    ~Builder() {
        delete[] queues;
    }

    void add_leaf(const IndexedLeaf& ileaf) {

        unsigned int leaf_count = ileaf.index - stream_index;
        while (leaf_count > 0) {
            /* Push invalid nodes so the index catches up with the stream. */
            unsigned int power = 0;

            if (leaf_count >= 8) {
                power = std::floor(std::log(leaf_count) / std::log(8));
            }

            unsigned int non_empty_i = queue_count;

            for (unsigned int i = 0; i < 8; i++) {
                if (!queues[i].empty()) {
                    non_empty_i = i;
                    break;
                }
            }

            unsigned int insert_i = std::min(power, non_empty_i);
            if (insert_i >= queue_count) {
                std::cout 
                << "\n\tERROR: completely homogenous model!" 
                << "\n\tConstruction aborted.\n\n\a";
                writer << ileaf.leaf;
                throw "Homogenous model";
            }

            Node to_push = ileaf.leaf.valid() ? Node(ileaf.leaf) : Node();
            push_at(to_push, insert_i);
            leaf_count -= std::pow(8, insert_i);
        } 

        stream_index = ileaf.index;
    }
};
