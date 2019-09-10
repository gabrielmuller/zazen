#pragma once

#include <cstdint>

struct Leaf {
    /* Leaf voxel. */

    uint8_t r, g, b, a;
    uint16_t n1, n2;

    Leaf() = default;

    Leaf(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
        r(r), g(g), b(b), a(a) {}

    Leaf(uint8_t v) : Leaf(v, v, v, v) {}

    inline void set_color(unsigned char* pixel, const float lightness) const {
        pixel[0] = r * lightness;
        pixel[1] = g * lightness;
        pixel[2] = b * lightness;
    }

    inline bool valid() const {
        return (bool) a;
    }

    inline bool operator!=(const Leaf& o) const {
        return r != o.r || g != o.g || b != o.b || a != o.a;
    }

    inline bool operator==(const Leaf& o) const {
        return !(*this != o);
    }
};
