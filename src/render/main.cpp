#include <GL/glut.h>
#include <GL/glut.h>
#include "render.cpp"

int t = 0;
void render_scene() {
    fflush(stdout);

    #pragma omp parallel for
    for (int  i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            render(texture[j][i], i, j, t);
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

    unsigned int arg = 1;
    if (arg >= argc) {
        std::cout << "Please specify an input file.\n";
        return 1;
    }

    block = from_file(argv[arg]);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    glutInitWindowPosition(100, 100);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow(" ");

    glutDisplayFunc(render_scene);
    glutIdleFunc(render_scene);
    glutMainLoop();

    return 0;
}
