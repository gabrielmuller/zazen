#include <GL/glut.h>
#include <GL/glut.h>
#include <math.h>

#define WIDTH 1024
#define HEIGHT 1024

unsigned char texture[WIDTH][HEIGHT][3];             
static int t = 0;

void render(unsigned char* pixel, int i, int j) {
    pixel[0] = (i + j + t) % 0x100;
    pixel[1] = j * t % 0x100;
    pixel[2] = i * j - t % 0x100;
    t++;
}

void renderScene() {    

    // render the texture here

    printf("Helheorhor");
    for (int  i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            unsigned char* pixel = texture[i][j];
            render(pixel, i, j);
        }
    }

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
