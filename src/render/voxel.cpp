#pragma once

#include <cstdint>

struct Voxel {
    /* Non-leaf voxel. */

    uint32_t child;
    uint8_t valid; // 8 flags of whether children are visible
    uint8_t leaf;  // 8 flags of whether children are leaves

    int32_t address_of(uint8_t octant) {
        /* Get address in block of child octant. */
        uint8_t mask = ~(0xff << octant);
        return child + __builtin_popcount(mask & valid);
    }

};
