#pragma once

#include "vector.cpp"
const float e = std::numeric_limits<float>::epsilon();
struct Ray {
    /* Simple container class for a ray. */
    Vector origin, direction, negdir;

    Ray() : origin(Vector()), direction(Vector()) {}

    explicit Ray(const Vector& origin, Vector& direction) :
        origin(origin), init_origin(origin) {
            if (fabsf(direction.x) < e) direction.x = copysignf(e, direction.x);
            if (fabsf(direction.y) < e) direction.y = copysignf(e, direction.y);
            if (fabsf(direction.z) < e) direction.z = copysignf(e, direction.z);
            this->direction = direction;
    }

    uint8_t octant_mask() {
        uint8_t mask = 0;
        if (direction.x >= 0) mask ^= 4;
        if (direction.y >= 0) mask ^= 2;
        if (direction.z >= 0) mask ^= 1;
        return mask;
    }

    Ray march(float amount) {
        Vector diff(direction.x * amount,
                    direction.y * amount,
                    direction.z * amount); 

        origin += diff;
        return *this;
    }

    inline float square_distance() {
        return (origin - init_origin).squared();
    }

  private:
    Vector init_origin;
};

