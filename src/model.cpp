#pragma once

struct Model {
    virtual Leaf get(unsigned int x, unsigned int y, unsigned int z) = 0;
    virtual ~Model() {}
};
