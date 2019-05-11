#include <cmath>
#include <new>
#include <limits>
#include <cstring>

#include "leaf.cpp"
#include "ray.cpp"
#include "stack.cpp"
#include "block.cpp"

const unsigned int WIDTH = 300;
const unsigned int HEIGHT = 300;

const float fov = 1.2; // 1 -> 90 degrees
const Vector scale(1, 1, 316/512.0);
unsigned char texture[WIDTH][HEIGHT][3];             

Block* block = nullptr;

void render(unsigned char* pixel, int i, int j, int t) {
    const float screen_x = (i * fov) / (float) WIDTH - 0.5;
    const float screen_y = (j * fov) / (float) HEIGHT - 0.5;

    const float time = t / 60.0F;

    // start traverse on root voxel
    Vector origin(sin(time)*0.9, sin(time/3.21) * 0.9 + 1.1, cos(time/1.12)*0.9);
    Vector direction(screen_x * scale.x, -1 * scale.y, screen_y * scale.z);

    Ray ray(origin, direction);

    VoxelStack stack(20, 2.0);
    stack.push_root(&block->get<Voxel>(0), Vector(-1, -1, -1), ray);

    pixel[0] = 0x33;
    pixel[1] = 0x33;
    pixel[2] = 0x00;

    while (true) {

        const uint8_t oct = stack->octant;
        bool valid = (stack->voxel->valid >> oct) & 1;
        bool leaf = (stack->voxel->leaf >> oct) & 1;
        if (leaf) {
            /* Ray origin is inside leaf voxel, render leaf. */
            Leaf* leaf = &block->get<Leaf>(stack->voxel->address_of(oct));
            ray.distance *= 0.2;
            ray.distance += 1;
            float lightness = 1/(ray.distance * ray.distance);
            leaf->set_color(pixel, lightness);
            break;
        } 

        if (valid) {
            /* Go a level deeper. */
            stack.push(&block->get<Voxel>(stack->voxel->address_of(oct)), ray);

            pixel[1] += (0xff - pixel[1]) / 8;

        } else {
            /* Ray origin is in invalid voxel, cast ray until it hits next
             * voxel. 
             */
            Vector child_corner(stack->corner);

            float child_size = stack.box_size * 0.5;
            child_corner.adjust_corner(child_size, oct);
            uint8_t mask = ray.octant_mask();

            Vector mirror_origin = ray.origin.mirror(mask);
            Vector mirror_direction = ray.direction.mirror(mask);
            Vector mirror_corner = child_corner.mirror(mask);
            mirror_corner.adjust_corner(-child_size, mask);

            float tx = (mirror_corner.x - mirror_origin.x) / mirror_direction.x;
            float ty = (mirror_corner.y - mirror_origin.y) / mirror_direction.y;
            float tz = (mirror_corner.z - mirror_origin.z) / mirror_direction.z;
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

            /* Ray will start next step at the point of this intersection */
            ray.march(t);

            while (hit_face & ~(stack->octant ^ mask) && !stack.empty()) {
                /* Hit face is at this voxel's boundary, search parent */
                stack.pop();
            }

            if (stack.empty()) {
                /* Ray is outside root octree. */

                // EMPTY STACK AFTER LOOP - BLUE
                pixel[2] = 0xc0;
                break;
            }
            /* Loop end: found ancestral voxel with space on the hit axis.
             * Transfer to sibling voxel, changing on the axis of the face
             * that was hit.
             */
            stack->octant ^= hit_face;
        }
    }
}
