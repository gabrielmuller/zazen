#pragma once

#include <cstdint>

struct Leaf {
    /* Leaf voxel. */

    uint8_t r, g, b, a;

    Leaf() = default;

    Leaf(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
        r(r), g(g), b(b), a(a) {}

    inline void set_color(unsigned char* pixel, float lightness) const {
        pixel[0] = r * lightness;
        pixel[1] = g * lightness;
        pixel[2] = b * lightness;
    }
};
