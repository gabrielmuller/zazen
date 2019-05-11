#pragma once

#include <cstdint>

struct Voxel {
    /* Non-leaf voxel. */

    int32_t child;
    uint8_t valid; // 8 flags of whether children are visible
    uint8_t leaf;  // 8 flags of whether children are leaves

    size_t address_of(uint8_t octant) {
        /* Get address in block of child octant. */
        size_t address = child;
        for (int i = 0; i < octant; i++) {
            if ((1 << i) & (valid)) address++;
        }
        return address;
    }

};
