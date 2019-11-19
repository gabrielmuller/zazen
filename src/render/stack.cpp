#pragma once

#include "voxel.cpp"
#include "vector.cpp"

struct StackEntry {
    Voxel* voxel;   // pointer to voxel in Block
    uint8_t octant; // octant ray origin is in this voxel
    Vector corner;  // inferior corner
};

struct VoxelStack {
    StackEntry* entries;
    size_t top;
    float box_size;

    explicit VoxelStack(const size_t size, const float init_box_size) {
        entries = new StackEntry[size];
        top = 0;
        box_size = init_box_size;
    }

    ~VoxelStack() {
        delete[] entries;
    }

    void push(Voxel* voxel, Ray ray) {
        entries[top] = {voxel, 0, peek().corner};
        box_size *= 0.5;
        entries[top].corner.adjust_corner(box_size, peek().octant);
        top++;
        peek().octant = get_octant(ray);
    }

    void push_root(Voxel* voxel, Vector corner, Ray ray) {
        entries[top] = {voxel, 0, corner};
        top++;
        peek().octant = get_octant(ray);
    }

    inline void pop() {
        box_size *= 2;
        top--;
    }

    inline StackEntry* operator->() const {
        return entries + (top - 1);
    }

    inline StackEntry& peek() const {
        return *operator->();
    }

    inline bool empty() {
        return !top;
    }

    inline size_t size() {
        return top;
    }

    inline uint8_t get_octant (const Ray& ray) const {
        /* Returns which octant the vector resides inside box. */
        uint8_t octant = 0;
        const float oct_size = box_size * 0.5;
        const Vector& corner = peek().corner;

        if (ray.origin.x > corner.x + oct_size) octant ^= 4;
        if (ray.origin.y > corner.y + oct_size) octant ^= 2;
        if (ray.origin.z > corner.z + oct_size) octant ^= 1;
        return octant;
    }
};
