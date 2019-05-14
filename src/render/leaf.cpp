#pragma once

#include <cstdint>

struct Leaf {
    /* Leaf voxel. */
    static float transparency;

    uint8_t r, g, b, a;
    uint16_t n1, n2;

    Leaf() = default;

    Leaf(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
        r(r), g(g), b(b), a(a) {}

    inline void set_color(unsigned char* pixel, float distance) const {
        const float alpha = transparency * a;
        //pixel[0] = clamp(pixel[0] + alpha * distance);
        pixel[0] = clamp(pixel[0] + alpha * distance * 0.9);
        pixel[1] = clamp(pixel[1] + alpha * distance);
        pixel[2] = clamp(pixel[2] + alpha * distance);
    }

    inline bool operator!=(Leaf& o) const {
        return r != o.r || g != o.g || b != o.b || a != o.a;
    }

  private:
    inline uint8_t clamp(unsigned int i) const {
        if (i > 0xff) return 0xff;
        return i;
    }
};

float Leaf::transparency = 0.01;
