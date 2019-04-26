#include <GL/glut.h>
#include <GL/glut.h>
#include <cmath>
#include <stdio.h>
#include <new>
#include <limits>
#include <cstring>

#include "leaf.cpp"
#include "ray.cpp"
#include "stack.cpp"
#include "block.cpp"

#define WIDTH 512
#define HEIGHT 512

const float fov = 1; // 1 -> 90 degrees
unsigned char texture[WIDTH][HEIGHT][3];             
int t = 0;

Block* block = nullptr;

void render(unsigned char* pixel, int i, int j) {
    bool do_log = !((i+1)%(WIDTH/8))&&!((j+1)%(HEIGHT/8));
    do_log |= !i && !j;


    const float screen_x = (i * fov) / (float) WIDTH - 0.5;
    const float screen_y = (j * fov) / (float) HEIGHT - 0.5;

    const float time = t / 60.0F;

    // start traverse on root voxel
    Vector origin(sin(time)*0.999, cos(time)*0.999, sin(time/2)-2);
    Vector direction(screen_x, screen_y, 1);

    Ray ray(origin, direction);

    VoxelStack stack(20, 2.0);
    stack.push_root(&block->get<Voxel>(0), Vector(-1, -1, -1), ray);

    pixel[0] = 0x33;
    pixel[1] = 0x33;
    pixel[2] = 0x00;

    if (do_log) printf("\n\n\n*************\n* NEW FRAME *\n*************\n\n");

    while (true) {

        const uint8_t oct = stack->octant;
        bool valid = (stack->voxel->valid >> oct) & 1;
        bool leaf = (stack->voxel->leaf >> oct) & 1;
        if (leaf) {
            /* Ray origin is inside leaf voxel, render leaf. */
            Leaf* leaf = &block->get<Leaf>(stack->voxel->address_of(oct));
            // XXX LEAF - RED
            //if (ray.distance < 1.01) ray.distance = 1.01;
            if (do_log) printf("Distance: %f\n", ray.distance);
            ray.distance *= 0.5;
            ray.distance += 1;
            float lightness = 2/(ray.distance * ray.distance);
            leaf->set_color(pixel, lightness);
            if (do_log)
            printf("Leaf painted %x %x %x\n", leaf->r, leaf->g, leaf->b);
            break;
        } 

        if (valid) {
            /* Go a level deeper. */
            stack.push(&block->get<Voxel>(stack->voxel->address_of(oct)), ray);

            if (do_log) printf("\nGo deeper\n"), stack.print();
            // XXX EVERY MARCH - SLIGHTLY GREENER
            pixel[1] += 0x30;

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
            if (!hit_face) {
                // XXX NO FACE HIT - GREEN
                pixel[1] = 0xff;
            }

            if (do_log){
            printf("Pixel (x, y):  (%d, %d)\n", i, j);
            printf("Child box corn:");
            child_corner.print();
            printf("Box corner:    ");
            stack->corner.print();
            printf("stack-> size:      %f\n", stack.box_size);
            printf("Ray direction: ");
            ray.direction.print();
            printf("Mirror directi:");
            mirror_direction.print();
            printf("Ray origin:    ");
            ray.origin.print();
            printf("Mirror origin: ");
            mirror_origin.print();
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
                printf("Before: ");
                stack.print();
                printf("hit&~(oct^mask)=%d\n", hit_face & ~(stack->octant ^ mask));
            }
            while (hit_face & ~(stack->octant ^ mask) && !stack.empty()) {
                if (do_log)
                printf("Peek oct:       %d\n", stack->octant);
                /* Hit face is at this voxel's boundary, search parent */
                stack.pop();
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
            stack->octant ^= hit_face;
            if (do_log) {
                printf("After:  ");
                stack.print();
            }
        }
    }
}

void renderScene() {    

    // render the texture here

    fflush(stdout);

    #pragma omp parallel for
    for (int  i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            render(texture[j][i], i, j);
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
    block = new Block(11);
    Voxel* p = new (block->slot()) Voxel();
    Voxel* c = new (block->slot()) Voxel();
    new (block->slot()) Leaf(0xff, 0x00, 0xff, 0xff);
    Voxel* d = new (block->slot()) Voxel();
    new (block->slot()) Leaf(0xff, 0x88, 0x11, 0xff);
    new (block->slot()) Leaf(0xff, 0xff, 0x00, 0xff);
    new (block->slot()) Leaf(0x00, 0xff, 0x80, 0xff);
    Voxel* e = new (block->slot()) Voxel();
    new (block->slot()) Leaf(0xff, 0x33, 0x33, 0xff);
    new (block->slot()) Leaf(0xff, 0xaa, 0xaa, 0xff);
    new (block->slot()) Leaf(0xff, 0xff, 0xff, 0xff);

    p->child = 1;
    p->valid = 0x8a;
    p->leaf = 0x08;

    c->child = 4;
    c->valid = 0x82;
    c->leaf = 0x82;

    d->child = 6;
    d->valid = 0x82;
    d->leaf = 0x02;

    e->child = 8;
    e->valid = 0xa2;
    e->leaf = 0xa2;

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

