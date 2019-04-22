#include <GL/glut.h>
#include <GL/glut.h>
#include <cmath>
#include <stdio.h>
#include <new>
#include <limits>
#include <cstring>

#define WIDTH 256
#define HEIGHT 256

const float e = std::numeric_limits<float>::epsilon() * 1;
const float fov = 1; // 1 -> 90 degrees
unsigned char texture[WIDTH][HEIGHT][3];             
int t = 0;

struct Voxel {
    /* Non-leaf voxel. */

    // 15 high bits: relative child pointer
    // 1 lowest bit: far flag TODO
    uint16_t child;
    uint8_t valid; // 8 flags of whether children are visible
    uint8_t leaf;  // 8 flags of whether children are leaves

    size_t address_of(uint8_t oct) {
        /* Get address in block of child octant. */
        size_t address = child;
        for (int i = 0; i < oct; i++) {
            if ((1 << i) & (valid | leaf)) address++;
        }
        return address;
    }
};

struct Leaf {
    /* Leaf voxel. */

    uint8_t r, g, b, a;

    Leaf() = default;

    Leaf(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
        r(r), g(g), b(b), a(a) {}

    inline void set_color(unsigned char* pixel, float lightness) const {
        pixel[0] = r * lightness;
        pixel[1] = g * lightness;
        pixel[2] = b * lightness;
    }
};
struct Vector {
    /* Simple container struct for a 3D vector. */
    float x, y, z;
    Vector() = default;
    explicit Vector(float x, float y, float z) :
        x(x), y(y), z(z) {}

    Vector(const Vector& v) : Vector(v.x, v.y, v.z) {}

    inline float magnitude() const {
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

inline void printv(Vector vector) {
    printf("(%f, %f, %f) %f\n", vector.x, vector.y, vector.z, vector.magnitude());
}

struct Ray {
    /* Simple container class for a ray. */
    Vector origin, direction, negdir;
    float distance;

    explicit Ray(const Vector origin, Vector direction) :
        origin(origin), distance(0) {
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

        origin.x += diff.x;
        origin.y += diff.y;
        origin.z += diff.z;
        distance += diff.magnitude();

        return *this;
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
        (Ray ray) const {
            /* Returns which octant the vector resides inside box. */
            uint8_t octant = 0;
            const float oct_size = size * 0.5;

            // If point is at border, adjust according to ray direction
            while (ray.origin.x == corner.x + oct_size ||
                   ray.origin.y == corner.y + oct_size ||
                   ray.origin.z == corner.z + oct_size) {
                ray.march(e * 100);
            }

            if (ray.origin.x > corner.x + oct_size) octant ^= 4;
            if (ray.origin.y > corner.y + oct_size) octant ^= 2;
            if (ray.origin.z > corner.z + oct_size) octant ^= 1;
            return octant;
        }
};

struct StackEntry {
    Voxel* voxel;   // pointer to voxel in Block
    uint8_t octant; // octant ray origin is in this voxel
    Vector corner;  // inferior corner
};

struct VoxelStack {
    StackEntry* entries;
    size_t top;

    explicit VoxelStack(const size_t size) {
        entries = new StackEntry[size];
        top = 0;
    }

    ~VoxelStack() {
        delete[] entries;
    }

    void push(Voxel* voxel, uint8_t octant, Vector corner) {
        entries[top] = {voxel, octant, corner};
        top++;
    }

    StackEntry& pop() {
        /* call peek_oct before this if you need the octant position */
        top--;
        return entries[top];
    }

    inline StackEntry& peek() {
        return entries[top - 1];
    }

    inline bool empty() {
        return !top;
    }

    inline size_t size() {
        return top;
    }

    inline void print() {
        for (int i = 0; i < top; i++) printf("%d ", entries[i].octant);
        printf("\n");
    }
};

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
    bool do_log = !(i%32)&&!(j%32);


    const float screen_x = (i * fov) / (float) WIDTH - 0.5;
    const float screen_y = (j * fov) / (float) HEIGHT - 0.5;

    const float time = t / 60.0F;

    // start traverse on root voxel
    AABB box(Vector(-1, -1, -1), 2);
    Ray ray(Vector(sin(time*2)*0.6, 0.9, -0.9), Vector(screen_x, screen_y, 1).normalized());
    VoxelStack stack(20);
    stack.push(&block->get<Voxel>(0),
                box.get_octant(ray),
                box.corner
                );

    pixel[0] = 0x00;
    pixel[1] = 0x00;
    pixel[2] = 0x00;

    if (do_log) printf("\n\n\n*************\n* NEW FRAME *\n*************\n\n");
    if (do_log) printf("Debug pixel (%d, %d)\n", i / 32, j / 32);

    while (true) {

        const uint8_t oct = stack.peek().octant;
        bool valid = (stack.peek().voxel->valid >> oct) & 1;
        bool leaf = (stack.peek().voxel->leaf >> oct) & 1;
        if (leaf) {
            /* Ray origin is inside leaf voxel, render leaf. */
            Leaf* leaf = &block->get<Leaf>(stack.peek().voxel->address_of(oct));
            // XXX LEAF - RED
            if (ray.distance < 1.01) ray.distance = 1.01;
            if (do_log) printf("Distance: %f\n", ray.distance);
            float lightness = 1/(ray.distance * ray.distance);
            leaf->set_color(pixel, lightness);
            if (do_log)
            printf("Leaf painted %x %x %x\n", leaf->r, leaf->g, leaf->b);
            return;
        } 

        if (valid) {
            /* Go a level deeper. */

            // Adjust inferior corner
            box.size *= 0.5;
            if (oct & 4) box.corner.x += box.size;
            if (oct & 2) box.corner.y += box.size;
            if (oct & 1) box.corner.z += box.size;

            stack.push(&block->get<Voxel>(stack.peek().voxel->address_of(oct)),
                        box.get_octant(ray),
                        box.corner);

            if (do_log) printf("\nGo deeper, stack size now %lu\n", stack.size());
            // XXX EVERY MARCH - SLIGHTLY GREENER
            pixel[1] += 0x20;

        } else {
            /* Ray origin is in invalid voxel, cast ray until it hits next
             * voxel. 
             */

            StackEntry child = {
                nullptr,
                box.get_octant(ray),
                box.corner
            };

            float child_size = box.size * 0.5;

            if (oct & 4) child.corner.x += child_size;
            if (oct & 2) child.corner.y += child_size;
            if (oct & 1) child.corner.z += child_size;

            Vector centered(ray.origin.x - (child.corner.x + child_size * 0.5), 
                            ray.origin.y - (child.corner.y + child_size * 0.5), 
                            ray.origin.z - (child.corner.z + child_size * 0.5)  
            );

            uint8_t mask = ray.octant_mask();
            Vector mirror_direction = ray.direction.mirror(mask);
            Vector mirror_origin = centered.mirror(mask);
            if (do_log) {
                printf("Mask %d, Center mirror: ", mask); printv(mirror_origin);
            }

            mirror_origin.x += (child.corner.x + child_size * 0.5);
            mirror_origin.y += (child.corner.y + child_size * 0.5);
            mirror_origin.z += (child.corner.z + child_size * 0.5);

            float tx = (child.corner.x - mirror_origin.x) / mirror_direction.x;
            float ty = (child.corner.y - mirror_origin.y) / mirror_direction.y;
            float tz = (child.corner.z - mirror_origin.z) / mirror_direction.z;
            float t = std::numeric_limits<float>::infinity();

            /* Detect which face hit. */
            uint8_t hit_face = 0;

            /* t is the smallest positive value of {tx, ty, tz} */
            if (tx > 0) {
                t = tx;
                hit_face = 4;
            }
            if (ty > 0 && ty < t) {
                t = ty;
                hit_face = 2;
            }
            if (tz > 0 && tz < t) {
                t = tz;
                hit_face = 1;
            }
            if (!hit_face) {
                // XXX NO FACE HIT - GREEN
                pixel[1] = 0xff;
            }

            if (do_log){
            printf("Pixel (x, y):  (%d, %d)\n", i, j);
            printf("Child box corn:");
            printv(child.corner);
            printf("Box corner:    ");
            printv(box.corner);
            printf("Box size:      %f\n", box.size);
            printf("Ray direction: ");
            printv(ray.direction);
            printf("Centered:      ");
            printv(centered);
            printf("Mirror directi:");
            printv(mirror_direction);
            printf("Ray origin:    ");
            printv(ray.origin);
            printf("Mirror origin: ");
            printv(mirror_origin);
            printf("Face hit:       %d\n", hit_face);
            printf("tx, ty, tz, t:  (%f, %f, %f) -> %f\n", tx, ty, tz, t);
            printf("Mask:           %d\n", mask);
            printf("Oct:            %d\n", oct);
            }

            /* Ray will start next step at the point of this intersection */
            t += std::numeric_limits<float>::epsilon();
            ray.march(t);

            if (do_log) {
                printf("-------------\n");
                printf("Before:size:    %lu\n", stack.size());
                printf("hit&~(oct^mask)=%d\n", hit_face & ~(stack.peek().octant ^ mask));
            }
            while (hit_face & ~(stack.peek().octant ^ mask)) {
                if (do_log)
                printf("Peek oct:       %d\n", stack.peek().octant);
                if (stack.empty()) {
                    /* Ray is outside root octree. */
                    // XXX EMPTY STACK - DARK BLUE
                    pixel[2] = 0x40;
                    break;
                }
                /* Hit face is at this voxel's boundary, search parent */
                stack.pop();
                box.size *= 2;
            }
            if (do_log) {
                printf("After:stk_size: %lu\n\n", stack.size());
            }

            if (stack.empty()) {
                /* Ray is outside root octree. */
                // XXX EMPTY STACK AFTER LOOP - BLUE
                pixel[2] = 0xc0;
                break;
            }
            /* Loop end: found ancestral voxel with space on the hit axis.
             * Transfer to sibling voxel, changing on the axis of the face
             * that was hit.
             */
            stack.peek().octant ^= hit_face;
        }
    }
    if (do_log) {
        pixel[0] = 0xff;
        pixel[1] = 0x00;
        pixel[2] = 0x00;
    }
    if (!(i / 32)) pixel[2] = 0xff;
    if (!(j / 32)) pixel[1] = 0xff;

}

void renderScene() {    

    // render the texture here

    fflush(stdout);
    for (int  i = 0; i < WIDTH/2; i++) {
        for (int j = 0; j < HEIGHT/2; j++) {
            unsigned char* pixel = texture[j*2][i*2];
            render(pixel, i, j);
            memcpy(texture[j*2+1][i*2], pixel, 3 * sizeof(char));
            memcpy(texture[j*2][i*2+1], pixel, 3 * sizeof(char));
            memcpy(texture[j*2+1][i*2+1], pixel, 3 * sizeof(char));
        }
    }

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
    block = new Block(8);
    Voxel* p = new (block->slot()) Voxel();
    Voxel* c = new (block->slot()) Voxel();
    Voxel* d = new (block->slot()) Voxel();
    new (block->slot()) Leaf(0xff, 0x00, 0xff, 0xff);
    new (block->slot()) Leaf(0xff, 0xff, 0x00, 0xff);
    new (block->slot()) Leaf(0xff, 0xff, 0x80, 0xff);
    new (block->slot()) Leaf(0xff, 0xff, 0xff, 0xff);

    p->child = 1;
    p->valid = 0x82;
    p->leaf = 0x00;

    c->child = 3;
    c->valid = 0x82;
    c->leaf = 0x82;

    d->child = 5;
    d->valid = 0x82;
    d->leaf = 0x82;

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

