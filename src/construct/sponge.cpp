#pragma once

#include "model.cpp"
#include <cmath>


struct SpongeModel : Model {
    SpongeModel(std::string name, 
            unsigned int width, unsigned int height, unsigned int depth) 
            : Model(name, width, height, depth) {}

    Leaf at(int3 pos) const override {
        const bool reduced_palette = true;
        bool filled = true;
        if (pos.x >= width || pos.y >= height || pos.z >= depth) return Leaf();
        for (unsigned int div = width / 3; div > 0; div /= 3) {
            if (((pos.x / div) % 3 == 1) +
                ((pos.y / div) % 3 == 1) +
                ((pos.z / div) % 3 == 1) > 1) {
                filled = false;
                break;
            }
        }
        return filled ? Leaf(0xff,
                             pos.x * 0xff / width,
                             pos.y * 0xff / height,
                             0xff) : Leaf();
    }

    ~SpongeModel() override {}
};
