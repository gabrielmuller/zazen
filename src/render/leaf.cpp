#pragma once

#include <cstdint>

struct Leaf {
    /* Leaf voxel. */

    uint8_t rgba[4];

    Leaf() = default;

    Leaf(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        : rgba{r, g, b, a} {}

    Leaf(uint8_t v) : Leaf(v, v, v, v) {}

    inline void set_color(unsigned char* pixel, const float lightness) const {
        pixel[0] = rgba[0] * lightness;
        pixel[1] = rgba[1] * lightness;
        pixel[2] = rgba[2] * lightness;
    }

    inline bool valid() const {
        return (bool) rgba[3];
    }

    inline bool operator!=(const Leaf& o) const {
        return (uint32_t) *rgba != (uint32_t) *o.rgba;
    }

    inline bool operator==(const Leaf& o) const {
        return !(*this != o);
    }
};
