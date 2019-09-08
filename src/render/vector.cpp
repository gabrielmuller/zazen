#pragma once

#include <cstdint>

struct Vector {
    /* Simple container struct for a 3D vector. */
    float x, y, z;
    Vector() = default;
    explicit Vector(float x, float y, float z) :
        x(x), y(y), z(z) {}

    Vector(const Vector& v) : Vector(v.x, v.y, v.z) {}

    inline float squared() const {
        return x*x + y*y + z*z;
    }

    Vector& normalized() {
        float invmag = 1 / sqrtf(squared());
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

    Vector operator+(const Vector& v) const {
        return Vector(v.x + x, v.y + y, v.z + z);
    }

    Vector operator-(const Vector& v) const {
        return Vector(v.x - x, v.y - y, v.z - z);
    }

    Vector operator*(const float scalar) const {
        return Vector(x * scalar, y * scalar, z * scalar);
    }

    Vector& operator+=(const Vector& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    Vector& operator-=(const Vector& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }



    void adjust_corner(float size, uint8_t octant) {
        if (octant & 4) x += size;
        if (octant & 2) y += size;
        if (octant & 1) z += size;
    }
};
