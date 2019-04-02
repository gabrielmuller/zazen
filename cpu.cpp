#include <GL/glut.h>
#include <GL/glut.h>
#include <cmath>
#include <stdio.h>
#include <new>
#include <limits>

#define WIDTH 128
#define HEIGHT 128

unsigned char texture[WIDTH][HEIGHT][3];             
int t = 0;

struct Voxel {
    /* Non-leaf voxel. */

    // 15 high bits: relative child pointer
    // 1 lowest bit: far flag TODO
    uint16_t child;
    uint8_t valid; // 8 flags of whether children are visible
    uint8_t leaf;  // 8 flags of whether children are leaves
};

struct Leaf {
    /* Leaf voxel. */

    uint8_t r, g, b, a;

    Leaf() = default;

    Leaf(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
            r(r), g(g), b(b), a(a) {}

     inline void set_color(unsigned char* pixel) const {
         pixel[0] = r;
         pixel[0] = g;
         pixel[0] = b;
    }
};
struct Vector {
    /* Simple container struct for a 3D vector. */
    float x, y, z;
    Vector() = default;
    explicit Vector(float x, float y, float z) :
            x(x), y(y), z(z) {}

    Vector(const Vector& v) : Vector(v.x, v.y, v.z) {}

    float magnitude() const {
        return sqrtf(x*x + y*y + z*z);
    }

    Vector& normalized() {
        float invmag = 1 / magnitude();
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
};

struct Ray {
    /* Simple container class for a ray. */
    Vector origin, direction, negdir;
    explicit Ray(const Vector origin, const Vector direction) :
            origin(origin), direction(direction) {
    }

    uint8_t octant_mask() {
        uint8_t mask = 0;
        if (direction.x >= 0) mask ^= 4;
        if (direction.y >= 0) mask ^= 2;
        if (direction.z >= 0) mask ^= 1;
        return mask;
    }
};

struct AABB {
    /* Axis-aligned bounding box */
    Vector corner; // bottom leftmost back corner
    float size;

    explicit AABB(const Vector corner, const float size) : 
            corner(corner), size(size) {}

    inline bool contains(const Vector vector) const 
    {
        /* Check if vector is inside box */
        return vector.x >= corner.x &&
               vector.y >= corner.y &&
               vector.z >= corner.z && 
               vector.x < corner.x + size &&
               vector.y < corner.y + size && 
               vector.z < corner.z + size;
    }

    inline uint8_t get_octant
            (const Vector vector) const {
        /* Returns which octant the vector resides inside box. */
        uint8_t octant = 0;
        float oct_size = size * 0.5;
        if (vector.x > corner.x + oct_size) octant ^= 4;
        if (vector.y > corner.y + oct_size) octant ^= 2;
        if (vector.z > corner.z + oct_size) octant ^= 1;
        return octant;
    }
};

struct VoxelStack {
    Voxel** voxels;
    uint8_t* octants;
    size_t top;

     explicit VoxelStack(const size_t size) {
        voxels = new Voxel*[size];
        octants = new uint8_t[size];
        top = 0;
    }

     ~VoxelStack() {
        delete[] voxels;
        delete[] octants;
    }

     void push(Voxel* voxel, const uint8_t octant) {
        voxels[top] = voxel;
        octants[top] = octant;
        top++;
    }

     Voxel* pop() {
        /* call peek_oct before this if you need the octant position */
        top--;
        return voxels[top];
    }

     inline Voxel* peek() {
        return voxels[top - 1];
    }

     inline uint8_t& peek_oct() {
        return octants[top - 1];
    }

     inline bool empty() {
        return !top;
    }
        
};

inline void printv(Vector vector) {
    printf("(%f, %f, %f) %f\n", vector.x, vector.y, vector.z, vector.magnitude());
}

struct Block {
    static const std::size_t element_size = 4;
    const size_t element_count;
    char* data = nullptr;
    char* front = nullptr;

    explicit Block(size_t element_count, char* data) :
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
    T& get(const std::size_t index) const {
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

Block* block = nullptr;

void render(unsigned char* pixel, int i, int j) {
    const float screen_x = i / (float) WIDTH - 0.5;
    const float screen_y = j / (float) HEIGHT - 0.5;

    const float time = t / 60.0F;

    // start traverse on root voxel
    VoxelStack stack(20);
    stack.push(&block->get<Voxel>(0), 0);
    AABB box(Vector(-1, -1, -1), 2);
    Ray ray(Vector(0,0,0), Vector(sin(time)+screen_x, cos(time)+screen_y, cos(time)).normalized());
    stack.peek_oct() = box.get_octant(ray.origin);
    float distance = 0;

    pixel[0] = 0;
    pixel[1] = 0;
    pixel[2] = 0;


    while (true) {
        if (stack.empty()) {
            pixel[1] = 0xaf;
            pixel[2] = 0xaf;
            break;
        }

        uint8_t oct = stack.peek_oct();
        bool valid = (stack.peek()->valid >> oct) & 1;
        bool leaf = (stack.peek()->leaf >> oct) & 1;
        if (leaf) {
            /* Ray origin is inside leaf voxel, render leaf. */
            Leaf* leaf = &block->get<Leaf>(stack.peek()->child + oct);
            pixel[0] = 0xff - distance;
            break;
        } else if (valid) {
            /* Go a level deeper. */
            stack.push(&block->get<Voxel>(stack.peek()->child + oct), oct);
            box.size *= 0.5;
            pixel[0] += 0x20;
        } else {
            /* Ray origin is in invalid voxel, cast ray until it hits next
             * voxel. 
             */
            AABB child(box.corner, box.size * 0.5);

            if (oct & 4) child.corner.x += child.size;
            if (oct & 2) child.corner.y += child.size;
            if (oct & 1) child.corner.z += child.size;

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
            float t = std::numeric_limits<float>::infinity();

            /* Detect which face hit. */
            uint8_t hit_face = 0;
            bool direction;

            /* t is the smallest positive value of {tx, ty, tz} */
            if (tx >= 0) {
                t = tx;
                hit_face = 4;
            }
            if (ty >= 0 && ty < t) {
                t = ty;
                hit_face = 2;
            }
            if (tz >= 0 && tz < t) {
                t = tz;
                hit_face = 1;
            }
            if (!hit_face) {
                pixel[2] = 0x8f;
                printf("BLUE\n");
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
                return;
            } else pixel[2] = hit_face * 10;

            if (!i && !j){
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
            t += std::numeric_limits<float>::epsilon();
            ray.origin = Vector(
                t * ray.direction.x + ray.origin.x,
                t * ray.direction.y + ray.origin.y,
                t * ray.direction.z + ray.origin.z
            ).mirror(mask);

            while (hit_face & (stack.peek_oct() ^ mask)) {
                if (stack.empty()) {
                    /* Ray is outside root octree. */
                    pixel[1] = 0x8f;
                    return;
                }
                /* Hit face is at this voxel's boundary, search parent */
                stack.pop();
                box.size *= 2;
            }

            if (stack.empty()) {
                /* Ray is outside root octree. */
                pixel[1] = 0xaf;
                return;
            }
            /* Loop end: found ancestral voxel with space on the hit axis.
             * Transfer to sibling voxel, changing on the axis of the face
             * that was hit.
             */
            stack.peek_oct() = (stack.peek_oct() ^ hit_face);
        }

    }
}

void renderScene() {    

    // render the texture here

    fflush(stdout);
    for (int  i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            unsigned char* pixel = texture[i][j];
            render(pixel, i, j);
        }
    }

    printf("%d\n", t);
    t++;
    glEnable (GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D (
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        WIDTH,
        HEIGHT,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        texture
    );

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0,  1.0);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0,  1.0);
    glEnd();

    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

int main(int argc, char **argv) {
    block = new Block(3);
    Voxel* y = new (block->slot()) Voxel();
    y->child = 1;
    y->valid = 0x02;
    y->leaf = 0x02;
    new (block->slot()) Leaf(0xff, 0x00, 0xff, 0xff);
    new (block->slot()) Leaf(0xff, 0xff, 0x00, 0xff);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    glutInitWindowPosition(100, 100);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow(" ");

    glutDisplayFunc(renderScene);
    glutIdleFunc(renderScene);
    glutMainLoop();


    return 0;
}

