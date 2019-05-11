#pragma once

#include <cstdint>

struct Vector {
    /* Simple container struct for a 3D vector. */
    float x, y, z;
    Vector() = default;
    explicit Vector(float x, float y, float z) :
        x(x), y(y), z(z) {}

    Vector(const Vector& v) : Vector(v.x, v.y, v.z) {}

    inline float magnitude() const {
        return sqrtf(x*x + y*y + z*z);
    }

    Vector& normalized() {
        float invmag = 1 / magnitude();
        x *= invmag;
        y *= invmag;
        z *= invmag;
        return *this;
    }

    Vector mirror(uint8_t mask) const {
        float mirror_x = mask & 4 ? -x : x;
        float mirror_y = mask & 2 ? -y : y;
        float mirror_z = mask & 1 ? -z : z;
        return Vector(mirror_x, mirror_y, mirror_z);
    }

    inline void print() {
        printf("(%f, %f, %f) %f\n", x, y, z, magnitude());
    }

    void adjust_corner(float size, uint8_t octant) {
        if (octant & 4) x += size;
        if (octant & 2) y += size;
        if (octant & 1) z += size;
    }
};