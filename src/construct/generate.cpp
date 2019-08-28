#pragma once

#include "model.cpp"
#include <cmath>

struct GenerateModel : Model {
    GenerateModel(std::string name, 
            unsigned int width, unsigned int height, unsigned int depth) 
            : Model(name, width, height, depth) {}

    Leaf at(int3 pos) const override {
        const unsigned int mod = width / 16;
        const float xx = (pos.x%mod)*2 / (float) mod - 1;
        const float yy = (pos.y%mod)*2 / (float) mod - 1;
        const float zz = (pos.z%mod)*2 / (float) mod - 1;
        const float radius = (sin((pos.x/mod + pos.y/mod + pos.z/mod)/5)+1)*0.3;

        if (xx*xx + yy*yy + zz*zz > radius) {
            return Leaf(0);
        }

        const uint8_t value = 0x20;
        return Leaf(value*(pos.x/mod), value*(pos.y/mod), value*(pos.z/mod), 0xff);
    }

    ~GenerateModel() override {}
};
