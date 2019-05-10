#pragma once

#include <cstddef>
#include <iostream>
#include "../render/leaf.cpp"

struct Model {
    unsigned int width, height, depth;
    Model(std::string name, 
            unsigned int width, unsigned int height, unsigned int depth) 
            : width(width), height(height), depth(depth) {
        std::cout << "Model \"" << name << "\" read from file." << std::endl;
    }
    virtual Leaf get(unsigned int x, unsigned int y, unsigned int z) const = 0;
    virtual ~Model() {}
};
