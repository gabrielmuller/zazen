#include "book/book.h"
#include "book/gpu_anim.h"
#include <cmath>

#define _ HANDLE_ERROR

/* Three bits, for example: BOTTOM_RIGHT_FRONT
 * 0           1           1 
 * X           Y           Z
 * LEFT/RIGHT  BOTTOM/TOP  BACK/FRONT
 */
enum Octant {0: LEFT_BOTTOM_BACK,
             1: LEFT_BOTTOM_FRONT,
             2: LEFT_TOP_BACK,
             3: LEFT_TOP_FRONT,
             4: RIGHT_BOTTOM_BACK,
             5: RIGHT_BOTTOM_FRONT,
             6: RIGHT_TOP_BACK,
             7: RIGHT_TOP_FRONT
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
    float x, y, z;
    __device__ explicit Vector(float x, float y, float z) :
            x(x), y(y), z(z) {}


            

};

__device__ struct Ray {
    /* Simple container class for a ray. */
    Vector origin, direction, negdir;
    Octant octant_mask;
    __device__ explicit Ray(const Vector origin, const Vector direction) :
            origin(origin), direction(direction) {
        float x = origin.x > 0 ? -origin.x : origin.x;
        float y = origin.y > 0 ? -origin.y : origin.y;
        float z = origin.z > 0 ? -origin.z : origin.z;
        negdir = Vector(x, y, z);

        // calculations are simpler if ray direction is negative in all axis,
        // so coordinate system is adjusted accordingly.
        octant_mask = RIGHT_TOP_FRONT;
        if (ray.direction.x > 0) octant_mask ^= 4;
        if (ray.direction.y > 0) octant_mask ^= 2;
        if (ray.direction.z > 0) octant_mask ^= 1;
    }

};

__device__ struct AABB {
    /* Axis-aligned bounding box */
    Vector corner; // bottom leftmost back corner
    float size;

    __device__ explicit AABB(const corner, const float size) : 
            corner(corner), size(size) {}

    __device__ inline bool inside(const Vector vector) const 
    {
        /* Check if vector is inside box */
        return vector.x >= corner.x &&
               vector.y >= corner.y &&
               vector.z >= corner.z && 
               vector.x < corner.x + size &&
               vector.y < corner.y + size && 
               vector.z < corner.z + size;
    }

    __device__ inline Octant get_octant
            (Vector vector) const {
        /* Returns which octant the vector resides inside box. */
        size *= 0.5;
        if (inside(corner, size)) {
            return LEFT_BOTTOM_BACK;
        } else if (vector.inside(Vector(corner.x, corner.y + size, corner.z),
                                 size)) {
            return LEFT_TOP_BACK;
        } else if (vector.inside(Vector(corner.x, corner.y, corner.z + size),
                                 size)) {
            return LEFT_BOTTOM_FRONT;
        } else if (vector.inside(
                          Vector(corner.x, corner.y + size, corner.z + size),
                          size)) {
            return LEFT_TOP_FRONT;
        }
        // The remaining options are all on the right
        corner.x += size;
        if (inside(corner, size)) {
            return RIGHT_BOTTOM_BACK;
        } else if (vector.inside(Vector(corner.x, corner.y + size, corner.z),
                                 size)) {
            return RIGHT_TOP_BACK;
        } else if (vector.inside(Vector(corner.x, corner.y, corner.z + size),
                                 size)) {
            return RIGHT_BOTTOM_FRONT;
        } else {
            return RIGHT_TOP_FRONT;
        }
    }
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
    Block(Block&& rhs) = delete; // No move constructor.
    Block& operator=(Block&& rhs) = delete; // No move assignment operator.

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

__device__ struct VoxelStack {
    Voxel** stack;
    size_t top;

    __device__ explicit VoxelStack(const size_t size) {
        stack = new Voxel*[size];
        top = 0;
    }

    __device__ ~VoxelStack() {
        delete[] stack;
    }

    __device__ void push(const Voxel* voxel) {
        stack[top] = voxel;
        top++;
    }

    __device__ Voxel* pop() {
        top--;
        return stack[top];
    }
};

__global__ void set_global_block(char* data, size_t element_count) {
    block = new Block(element_count, data);
}


__global__ void render(uchar4 *ptr, int ticks) {//,
                            //char* data, size_t element_count) {
    // map from threadIdx/BlockIdx to pixel position
    const int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = pixel_x + pixel_y * blockDim.x * gridDim.x;
    const float screen_x = pixel_x / (float) DIM - 0.5;
    const float screen_y = pixel_y / (float) DIM - 0.5;

    const float time = ticks / 60.0;

    // background are animated colors as placeholder
    ptr[offset].x = screen_x * (sin(time * 3) + 1) * 0.5 * 256;
    ptr[offset].y = (1 - screen_y) * (sin(time * 7) + 1) * 0.5 * 256;
    ptr[offset].z = (1 - screen_x) * screen_y * (cos(time * 13) + 1) * 128;
    ptr[offset].w = 0xff;

    // start traverse on root voxel
    VoxelStack stack(20);
    stack.push(&block->get<Voxel>(0));
    AABB box(Vector(-100, -100, -100), 200)
    Ray ray(Vector(0, 0, 0), Vector(screen_x, screen_y, 1.0));


    while (true) {
        Octant oct = box.get_octant(ray.origin);
        bool valid = (voxel->valid >> oct) & 1;
        bool leaf = (voxel->leaf >> oct) & 1;
        if (leaf) {
            // ray origin is inside leaf voxel, render leaf.
            Leaf* leaf = &block->get<Leaf>(voxel->children + oct);
            leaf->set_color(ptr[offset]);
            break;
        } else if (valid) {
            // go a level deeper
            voxel = &block->get<Voxel>(voxel->children + oct);
            box.voxel_size *= 0.5;
        } else {
            /* Ray origin is in invalid voxel, cast ray until it hits next
             * voxel. 
             * Since the coordinate system is mirrored and the ray's directions
             * are always negative, there are only three possible interior
             * faces to hit (back, left or bottom).
             */
            // detect which face hit
            float tx = (box.corner.x - ray.origin.x) / ray.negdir.x;
            float ty = (boy.corner.y - ray.origin.y) / ray.negdir.y;
            float tz = (boz.corner.z - ray.origin.z) / ray.negdir.z;
            float t;
            if (tx > ty && tx > tz) {
                t = tx;
            } else if (ty > tx && ty > tz) {
                t  = ty;
            else {
                t = tz;
            }
            Vector intersection(t * ray.direction.x,
                                t * ray.direction.y,
                                t * ray.direction.z);
            ray.origin = intersection;
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
