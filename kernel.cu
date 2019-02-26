#include "book/book.h"
#include "book/gpu_anim.h"
#include <cmath>

#define _ HANDLE_ERROR
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
    char* data = nullptr;
    char* front = nullptr;

    __device__ explicit Block(size_t element_count, char* data) :
            element_count(element_count),
            data(data) {
        front = data;
    }

    explicit Block(size_t element_count) : element_count(element_count) {
        data = new char[element_count * element_size];
        front = data;
    }


    ~Block() {
        delete[] data;
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
    __device__ T& get(const std::size_t index) const {
        return ((T*) data)[index];
    }

    char* slot() {
        char* front_slot = front;
        front += element_size;
        return front_slot;
    }

    size_t size() {
        return element_count * element_size;
    }
};

__global__ void placeholder(uchar4 *ptr, int ticks) {
    // map from threadIdx/BlockIdx to pixel position
    const int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = pixel_x + pixel_y * blockDim.x * gridDim.x;
    const float screen_x = pixel_x / (float) DIM;
    const float screen_y = pixel_y / (float) DIM;

    const float time = ticks * 0.01;

    ptr[offset].x = screen_x * (sin(time * 3) + 1) * 0.5 * 256;
    ptr[offset].y = (1 - screen_y) * (sin(time * 7) + 1) * 0.5 * 256;
    ptr[offset].z = (1 - screen_x) * screen_y * (cos(time * 13) + 1) * 128;
    ptr[offset].w = 0xff;
}

__global__ void kernel(char* data, size_t element_count) {
    Block* block = new Block(element_count, data);
    printf("Here goes nothing: '");
    printf("%c", block->get<Voxel>(block->get<Voxel>(0).child + 1).valid);
    printf("'\n");
}


void generate_frame(uchar4 *pixels, void*, int ticks) {
    dim3    grids(DIM/16, DIM/16);
    dim3    threads(16, 16);
    placeholder<<<grids,threads>>>(pixels, ticks);
}

int main(void) {
    Block block(6);
    Voxel* y = new (block.slot()) Voxel();
    y->child = 1;
    y->valid = 'U';
    new (block.slot()) Leaf();
    Voxel* z = new (block.slot()) Voxel();
    z->child = 2;
    z->valid = 'f';
    new (block.slot()) Leaf();
    new (block.slot()) Leaf();
    new (block.slot()) Leaf();

    char* dev_data;
    _( cudaMalloc((void**) &dev_data, block.size()) );
    _( cudaMemcpy(dev_data, block.data, block.size(), cudaMemcpyHostToDevice) );


    kernel<<<1, 1>>>(dev_data, block.element_count);
    _( cudaFree(dev_data) );
    GPUAnimBitmap  bitmap(DIM, DIM, NULL);
    bitmap.anim_and_exit((void(*)(uchar4*,void*,int)) generate_frame, NULL);
}
