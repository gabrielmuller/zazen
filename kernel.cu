#include "book/book.h"
#include "book/gpu_anim.h"
#include <cmath>

const int DIM = 1024;

__global__ struct Voxel {
    /* Non-leaf voxel. */

    // 15 high bits: relative child pointer
    // 1 lowest bit: far flag TODO
    int16_t child;
    int8_t valid; // 8 flags of whether children are visible
    int8_t leaf;  // 8 flags of whether children are leaves
};

__global__ struct Leaf {
    /* Leaf voxel. */

    int8_t r, g, b, a;
};

__global__ struct Block {
    static const std::size_t element_size = 4;
    const size_t element_count;
    void* data = nullptr;
    void* front;

    explicit Block(size_t element_count) : element_count(element_count) {
        data = operator new(element_count * element_size);
        front = data;
    }

    ~Block() {
        operator delete(data);
    }

    Block(Block&) = delete; // No copy constructor.

    Block& operator=(Block&) = delete; // No assigning.

    // Move constructor
    Block(Block&& rhs) :
        element_count(rhs.element_count),
        data(rhs.data) {
        rhs.data = nullptr;
    }

    // Move assignment operator
    Block& operator=(Block&& rhs) {
        if (this != &rhs) {
            operator delete(data);
            data = rhs.data;
            rhs.data = nullptr;
        }
        return *this;
    }

    template <class T>
    T& get(std::size_t index) {
        return ((T*) data)[index];
    }

    template <class T>
    T* new_slot() {
        T* slot = (T*) front;
        front = (void*) (slot + 1);
        return slot;
    }
};

__global__ void kernel(uchar4 *ptr, int ticks) {
    // map from threadIdx/BlockIdx to pixel position
    const int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = pixel_x + pixel_y * blockDim.x * gridDim.x;
    const float screen_x = pixel_x /(float) DIM;
    const float screen_y = 1 -(pixel_y /(float) DIM);

    const float intensity = screen_x * screen_y *(sin(ticks * 0.01) + 1);

    const int color = intensity * 256;

    ptr[offset].x = color;
    ptr[offset].y = color;
    ptr[offset].z = color;
    ptr[offset].w = 0xff;
}


void generate_frame(uchar4 *pixels, void*, int ticks) {
    dim3    grids(DIM/16, DIM/16);
    dim3    threads(16, 16);
    kernel<<<grids,threads>>>(pixels, ticks);
}

int main(void) {
    Block block(6);
    Voxel y;
    Voxel z;

    y.child = 1;
    y.valid = 'U';
    z.child = 2;
    z.valid = 'f';

    *block.new_slot<Voxel>() = y;
    *block.new_slot<Leaf>() = Leaf();
    *block.new_slot<Voxel>() = z;
    *block.new_slot<Leaf>() = Leaf();
    *block.new_slot<Leaf>() = Leaf();
    *block.new_slot<Leaf>() = Leaf();

    std::cout << "Here goes nothing: '" <<
    block.get<Voxel>(block.get<Voxel>(0).child + 1).valid
    << "'\n";

    GPUAnimBitmap  bitmap(DIM, DIM, NULL);
    bitmap.anim_and_exit((void(*)(uchar4*,void*,int)) generate_frame, NULL);
}
