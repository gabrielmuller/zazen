#pragma once

#include <cstddef>
#include "../render/leaf.cpp"

struct Model {
    unsigned int width, height, depth;
    Model(unsigned int width, unsigned int height, unsigned int depth) 
            : width(width), height(height), depth(depth) {}
    virtual Leaf get(unsigned int x, unsigned int y, unsigned int z) const = 0;
    virtual ~Model() {}
};
