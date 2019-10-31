#pragma once

#include "model.cpp"
#include <cmath>

struct GenerateModel : Model {
    GenerateModel(std::string name, 
            unsigned int width, unsigned int height, unsigned int depth) 
            : Model(name, width, height, depth) {}

    Leaf get(unsigned int x, unsigned int y, unsigned int z) const override {
        const unsigned int mod = width / 4;
        const float xx = (x%mod)*2 / (float) mod - 1;
        const float yy = (y%mod)*2 / (float) mod - 1;
        const float zz = (z%mod)*2 / (float) mod - 1;
        const float radius = (sin((x/mod + y/mod + z/mod)/5)+1)*0.3;
        if ((xx*xx + yy*yy + zz*zz > radius) || (std::sin((x*16 + y*20 + z * 30) /(float)width) < 0)) {
            return Leaf(0,0,0,0);
        }

        const uint8_t value = 0x20;
        return Leaf(value*x/mod, value*y/mod, value*z/mod, 0xff);
    }

    ~GenerateModel() override {}
};
