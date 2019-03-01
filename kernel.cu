#include "book/book.h"
#include "book/gpu_anim.h"
#include <cmath>

#define _ HANDLE_ERROR

enum Octant {0: BOTTOM_LEFT_BACK,
             1: BOTTOM_LEFT_FRONT,
             2: BOTTOM_RIGHT_BACK,
             3: BOTTOM_RIGHT_FRONT,
             4: TOP_LEFT_BACK,
             5: TOP_LEFT_FRONT,
             6: TOP_RIGHT_BACK,
             7: TOP_RIGHT_FRONT
             }

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
    __device__ inline void set_color(uchar4 pixel) const {
        pixel.x = r;
        pixel.y = g;
        pixel.z = b;
        pixel.w = 0xff;
    }
};


__device__ struct Vector {
    /* Simple container struct for a 3D vector. */
    double x, y, z;
    __device__ explicit Vector(double x, double y, double z) :
            x(x), y(y), z(z) {}

    __device__ inline bool inside(const Vector corner, const double size) const 
    {
        /* Checks if a position is inside a cube defined by its smallest corner
         * and side length.
         */
        return ray.origin.x > corner.x &&
               ray.origin.y > corner.y &&
               ray.origin.z > corner.z && 
               ray.origin.x < corner.x + size &&
               ray.origin.y < corner.y + size && 
               ray.origin.z < corner.z + size;
    }

    __device__ inline Octant get_octant
            (Vector corner, double size) const {
        /* Returns get octant this vector resides on a cube. */
        size /= 2;
        if (inside(corner, size)) {
            return BOTTOM_LEFT_BACK;
        } else if (inside(Vector(corner.x, corner.y + size, corner.z), size)) {
            return TOP_LEFT_BACK;
        } else if (inside(Vector(corner.x, corner.y, corner.z + size), size)) {
            return BOTTOM_LEFT_FRONT;
        } else if (inside(Vector(corner.x, corner.y + size, corner.z + size),
                          size)) {
            return TOP_LEFT_FRONT;
        }
        // The remaining options are all on the right
        corner.x += size;
        if (inside(corner, size)) {
            return BOTTOM_RIGHT_BACK;
        } else if (inside(Vector(corner.x, corner.y + size, corner.z), size)) {
            return TOP_RIGHT_BACK;
        } else if (inside(Vector(corner.x, corner.y, corner.z + size), size)) {
            return BOTTOM_RIGHT_FRONT;
        } else {
            return TOP_RIGHT_FRONT;
        }
    }

            

};

__device__ struct Ray {
    /* Simple container class for a ray. */
    Vector origin, direction;
    __device__ explicit Ray(const Vector origin, const Vector direction) :
            origin(origin), direction(direction) {}

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
    Block(Block&& rhs) = delete; // No move constructor
    Block& operator=(Block&& rhs) = delete; // No move assignment operator

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

__device__ Block* block = nullptr;

__global__ void set_global_block(char* data, size_t element_count) {
    block = new Block(element_count, data);
}


__global__ void render(uchar4 *ptr, int ticks) {//,
                            //char* data, size_t element_count) {
    // map from threadIdx/BlockIdx to pixel position
    const int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = pixel_x + pixel_y * blockDim.x * gridDim.x;
    const double screen_x = pixel_x / (double) DIM - 0.5;
    const double screen_y = pixel_y / (double) DIM - 0.5;

    const double time = ticks / 60.0;

    // background are animated colors as placeholder
    ptr[offset].x = screen_x * (sin(time * 3) + 1) * 0.5 * 256;
    ptr[offset].y = (1 - screen_y) * (sin(time * 7) + 1) * 0.5 * 256;
    ptr[offset].z = (1 - screen_x) * screen_y * (cos(time * 13) + 1) * 128;
    ptr[offset].w = 0xff;

    // start traverse on root voxel
    Voxel voxel = block->get<Voxel>(0);
    Vector corner(-100, -100, -100); // bottom back left voxel corner
    double voxel_size = 200;
    Ray ray(Vector(0, 0, 0), Vector(screen_x, screen_y, 1.0));

    while (true) {
        Octant oct = ray.origin.get_octant(corner, voxel_size);
        bool valid = (voxel.valid >> oct) & 1;
        bool leaf = (voxel.leaf >> oct) & 1;
        if (leaf) {
            // ray origin is inside leaf voxel, render leaf
            Leaf leaf = block->get<Leaf>(voxel.children + oct);
            leaf.set_color(ptr[offset]);
            break;
        } else if (valid) {
            ...
        }
    }

}

__global__ void kernel() {
    printf("Here goes nothing: '");
    printf("%c", block->get<Voxel>(block->get<Voxel>(0).child + 1).valid);
    printf("'\n");
}


void generate_frame(uchar4 *pixels, void*, int ticks) {
    dim3    grids(DIM/16, DIM/16);
    dim3    threads(16, 16);
    render<<<grids,threads>>>(pixels, ticks);
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

    set_global_block<<<1, 1>>>(dev_data, block.element_count);
    kernel<<<1, 1>>>();

    GPUAnimBitmap bitmap(DIM, DIM, NULL);
    bitmap.anim_and_exit((void(*)(uchar4*,void*,int)) generate_frame, NULL);

    _( cudaFree(dev_data) );
}
