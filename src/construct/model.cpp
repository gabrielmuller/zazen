#pragma once

#include <cstddef>
#include <iostream>
#include "../render/leaf.cpp"

struct Model {
    const unsigned int width, height, depth;
    const std::string name;

    Model(std::string name, 
            unsigned int width, unsigned int height, unsigned int depth) 
            : width(width), height(height), depth(depth), name(name) {
        std::cout << "Reading model \"" << name << "\" ...\n";
    }
    virtual ~Model() {}

  //private:
    size_t index;

    virtual Leaf get(unsigned int x, unsigned int y, unsigned int z) const = 0;
    

};
