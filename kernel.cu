#include "book/book.h"
#include "book/gpu_anim.h"
#include <cmath>

#define _ HANDLE_ERROR

/* Three bits, for example: BOTTOM_RIGHT_FRONT
 * 0           1           1 
 * X           Y           Z
 * LEFT/RIGHT  BOTTOM/TOP  BACK/FRONT
 * Low values are ommited.
 */
enum Octant {LOWEST = 0,
             FRONT = 1,
             TOP = 2,
             TOP_FRONT = 3,
             RIGHT = 4,
             RIGHT_FRONT = 5,
             RIGHT_TOP = 6,
             RIGHT_TOP_FRONT = 7,
             };

const int DIM = 128;
__global__ struct Voxel {
    /* Non-leaf voxel. */

    // 15 high bits: relative child pointer
    // 1 lowest bit: far flag TODO
    uint16_t child;
    uint8_t valid; // 8 flags of whether children are visible
    uint8_t leaf;  // 8 flags of whether children are leaves
};

__global__ struct Leaf {
    /* Leaf voxel. */

    uint8_t r, g, b, a;

    Leaf() = default;

    Leaf(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
            r(r), g(g), b(b), a(a) {}

    __device__ inline void set_color(uchar4& pixel) const {
        pixel.x = r;
        pixel.y = g;
        pixel.z = b;
        pixel.w = 0xff;
    }
};


__device__ struct Vector {
    /* Simple container struct for a 3D vector. */
    float x, y, z;
    __device__ Vector() = default;
    __device__ explicit Vector(float x, float y, float z) :
            x(x), y(y), z(z) {}

    __device__ Vector(const Vector& v) : Vector(v.x, v.y, v.z) {}

    __device__ float magnitude() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __device__ Vector& normalized() {
        float invmag = 1 / magnitude();
        x *= invmag;
        y *= invmag;
        z *= invmag;
        return *this;
    }

    __device__ Vector mirror(uint8_t mask) const {
        float mirror_x = mask & 4 ? -x : x;
        float mirror_y = mask & 2 ? -y : y;
        float mirror_z = mask & 1 ? -z : z;
        return Vector(mirror_x, mirror_y, mirror_z);
    }
};

__device__ struct Ray {
    /* Simple container class for a ray. */
    Vector origin, direction, negdir;
    __device__ explicit Ray(const Vector origin, const Vector direction) :
            origin(origin), direction(direction) {
    }

    __device__ uint8_t octant_mask() {
        uint8_t mask = 0;
        if (direction.x > 0) mask ^= 4;
        if (direction.y > 0) mask ^= 2;
        if (direction.z > 0) mask ^= 1;
        return mask;
    }
};

__device__ struct AABB {
    /* Axis-aligned bounding box */
    Vector corner; // bottom leftmost back corner
    float size;

    __device__ explicit AABB(const Vector corner, const float size) : 
            corner(corner), size(size) {}

    __device__ inline bool contains(const Vector vector) const 
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
            (const Vector vector) const {
        /* Returns which octant the vector resides inside box. */
        uint8_t octant = (uint8_t) LOWEST;
        float oct_size = size * 0.5;
        if (vector.x > corner.x + oct_size) octant ^= 4;
        if (vector.y > corner.y + oct_size) octant ^= 2;
        if (vector.z > corner.z + oct_size) octant ^= 1;
        return (Octant) octant;
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
    Voxel** voxels;
    Octant* octants;
    size_t top;

    __device__ explicit VoxelStack(const size_t size) {
        voxels = new Voxel*[size];
        octants = new Octant[size];
        top = 0;
    }

    __device__ ~VoxelStack() {
        delete[] voxels;
        delete[] octants;
    }

    __device__ void push(Voxel* voxel, const Octant octant) {
        voxels[top] = voxel;
        octants[top] = octant;
        top++;
    }

    __device__ Voxel* pop() {
        /* call peek_oct before this if you need the octant position */
        top--;
        return voxels[top];
    }

    __device__ inline Voxel* peek() {
        return voxels[top - 1];
    }

    __device__ inline Octant& peek_oct() {
        return octants[top - 1];
    }

    __device__ inline bool empty() {
        return !top;
    }
        
};

__device__ inline void printv(Vector vector) {
    printf("(%f, %f, %f) %f\n", vector.x, vector.y, vector.z, vector.magnitude());
}
__global__ void set_global_block(char* data, size_t element_count) {
    block = new Block(element_count, data);
};


__global__ void render(uchar4 *ptr, int ticks) {//,
                            //char* data, size_t element_count) {
    // map from threadIdx/BlockIdx to pixel position
    const int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = pixel_x + pixel_y * blockDim.x * gridDim.x;
    const float screen_x = pixel_x / (float) DIM - 0.5;
    const float screen_y = pixel_y / (float) DIM - 0.5;

    const float time = ticks / 60.0F;

    // start traverse on root voxel
    VoxelStack stack(20);
    stack.push(&block->get<Voxel>(0), LOWEST);
    AABB box(Vector(-1, -1, -1), 2);
    Ray ray(Vector(0,0,0), Vector(sin(time)+screen_x, cos(time)+screen_y, cos(time)).normalized());
    stack.peek_oct() = box.get_octant(ray.origin);
    float distance = 0;

    ptr[offset].x = 0;
    ptr[offset].y = 0;
    ptr[offset].z = 0;
    ptr[offset].w = 0xff;


    while (true) {
        if (stack.empty()) {
            ptr[offset].y = 0xaf;
            ptr[offset].z = 0xaf;
            break;
        }

        Octant oct = stack.peek_oct();
        bool valid = (stack.peek()->valid >> oct) & 1;
        bool leaf = (stack.peek()->leaf >> oct) & 1;
        if (leaf) {
            /* Ray origin is inside leaf voxel, render leaf. */
            Leaf* leaf = &block->get<Leaf>(stack.peek()->child + oct);
            ptr[offset].x = 0xff - distance;
            break;
        } else if (valid) {
            /* Go a level deeper. */
            stack.push(&block->get<Voxel>(stack.peek()->child + oct), oct);
            box.size *= 0.5;
            ptr[offset].x += 0x20;
        } else {
            /* Ray origin is in invalid voxel, cast ray until it hits next
             * voxel. 
             */
            AABB child(box.corner, box.size * 0.5);

            if (oct & RIGHT) child.corner.x += child.size;
            if (oct & TOP)   child.corner.y += child.size;
            if (oct & FRONT) child.corner.z += child.size;

            Vector upper(child.corner.x + child.size,
                                child.corner.y + child.size,
                                child.corner.z + child.size);

            uint8_t mask = ray.octant_mask();
            Vector mirror_direction = ray.direction.mirror(mask);
            Vector mirror_origin = ray.origin.mirror(mask);
            Vector mirror_corner = child.corner.mirror(mask);
            float tx = (box.corner.x - mirror_origin.x) / mirror_direction.x;
            float ty = (box.corner.y - mirror_origin.y) / mirror_direction.y;
            float tz = (box.corner.z - mirror_origin.z) / mirror_direction.z;
            float t = INFINITY;

            /* Detect which face hit. */
            Octant hit_face = LOWEST;
            bool direction;

            // t is the smallest positive value of {tx, ty, tz}
            if (tx > 0) {
                t = tx;
                hit_face = RIGHT;
            }
            if (ty > 0 && ty < t) {
                t = ty;
                hit_face = TOP;
            }
            if (tz > 0 && tz < t) {
                t = tz;
                hit_face = FRONT;
            }
            if (!hit_face) {
                // what the heck?
                ptr[offset].z = 0x8f;
                return;
            } else ptr[offset].z = (uint8_t) hit_face * 10;

            if (!offset){
                printf("Mask:          %d\n", mask);
                printf("Box corner:    ");
                printv(box.corner);
                printf("Mirror corner: ");
                printv(mirror_corner);
                printf("Box size:      %f\n", box.size);
                printf("Ray direction: ");
                printv(ray.direction);
                printf("Mirror directi:");
                printv(mirror_direction);
                printf("Ray origin:    ");
                printv(ray.origin);
                printf("Face hit:       %d\n", hit_face);
                printf("tx, ty, tz, t:  (%f, %f, %f) -> %f\n", tx, ty, tz, t);
                printf("\n");
            }

            /* Ray will start next step at the point of this intersection */
            t *= 1.01;
            Ray diff = Vector(t * ray.direction.x,
                              t * ray.direction.y,
                              t * ray.direction.z);
                              //TODO
            ray.origin = Vector(t * ray.direction.x + ray.origin.x,
                          t * ray.direction.y + ray.origin.y,
                          t * ray.direction.z + ray.origin.z).mirror(mask);

            while (hit_face & (stack.peek_oct() ^ mask)) {
                if (stack.empty()) {
                    /* Ray is outside root octree. */
                    ptr[offset].y = 0x8f;
                    return;
                }
                /* Hit face is at this voxel's boundary, search parent */
                stack.pop();
                box.size *= 2;
            }

            if (stack.empty()) {
                /* Ray is outside root octree. */
                ptr[offset].y = 0xaf;
                return;
            }
            /* Loop end: found ancestral voxel with space on the hit axis.
             * Transfer to sibling voxel, changing on the axis of the face
             * that was hit.
             */
            stack.peek_oct() = (Octant)((uint8_t) stack.peek_oct() ^ hit_face);
        }

    }

}

__global__ void kernel() {
    printf("Here goes nothing: '");
    printf("%c", block->get<Voxel>(0).valid);
    printf("'\n");
}


void generate_frame(uchar4 *pixels, void*, int ticks) {
    dim3    grids(DIM/16, DIM/16);
    dim3    threads(16, 16);
    render<<<grids,threads>>>(pixels, ticks);
    cudaDeviceSynchronize();
}

int main(void) {
    Block block(3);
    Voxel* y = new (block.slot()) Voxel();
    y->child = 1;
    y->valid = 0xee;
    y->leaf = 0xee;
    new (block.slot()) Leaf(0xff, 0x00, 0xff, 0xff);
    new (block.slot()) Leaf(0xff, 0xff, 0x00, 0xff);

    char* dev_data;
    _( cudaMalloc((void**) &dev_data, block.size()) );
    _( cudaMemcpy(dev_data, block.data, block.size(), cudaMemcpyHostToDevice) );

    set_global_block<<<1, 1>>>(dev_data, block.element_count);


    GPUAnimBitmap bitmap(DIM, DIM, NULL);
    bitmap.anim_and_exit((void(*)(uchar4*,void*,int)) generate_frame, NULL);

    _( cudaFree(dev_data) );
}
