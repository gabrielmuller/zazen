#pragma once

#include "model.cpp"
#include <cmath>

struct GenerateModel : Model {
    GenerateModel(std::string name, 
            unsigned int width, unsigned int height, unsigned int depth) 
            : Model(name, width, height, depth) {}

    Leaf get(unsigned int x, unsigned int y, unsigned int z) const override {
        const unsigned int mod = width / 16;
        float xx = (x%mod)*2 / (float) mod - 1;
        float yy = (y%mod)*2 / (float) mod - 1;
        float zz = (z%mod)*2 / (float) mod - 1;
        float radius = (sin((x/mod + y/mod + z/mod)/5)+1)*0.3;
        uint8_t value = (xx*xx + yy*yy + zz*zz) < radius ? 0x20 : 0x00;
        return Leaf(value*(x/mod), value*(y/mod), value*(z/mod), (bool) value * 0xff);
    }

    ~GenerateModel() override {}
};
