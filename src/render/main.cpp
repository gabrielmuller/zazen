#define SDL_MAIN_HANDLED
#define GLEW_STATIC

#include <fstream>
#include <GL/glew.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "render.cpp"

unsigned char pixels[WIDTH][HEIGHT][4];

std::string read_file(const char* filename) {
    std::string content;
    std::ifstream stream(filename, std::ios::in);

    if(!stream.is_open()) {
        std::cerr << "Could not read file "   << filename 
                  << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while(!stream.eof()) {
        std::getline(stream, line);
        content.append(line + "\n");
    }

    stream.close();
    return content;
}

void position_camera(unsigned int tick) {
    const float time = tick / 60.0F;
    cam_center.origin = Vector(sin(time)*0.9,
                               sin(time/3.21) * 0.9 + 1.1,
                               cos(time/1.12)*0.9);
}

int main(int argc, char **argv) {
    const int arg = 1;

    if (arg >= argc) {
        std::cout << "Please specify an input file.\n";
        return 1;
    }

    block = from_file(argv[arg]);

    SDL_Init(SDL_INIT_VIDEO);
    atexit(SDL_Quit);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);


    SDL_Window* window = SDL_CreateWindow(
            "zazen",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            WIDTH, HEIGHT,
            SDL_WINDOW_OPENGL
    );

    SDL_GLContext context = SDL_GL_CreateContext(window);

    glewExperimental = GL_TRUE;
    glewInit();

    float vertices[] = {
        -1.0f, -1.0f, // Vertex 1 (X, Y)
         3.0f, -1.0f, // Vertex 2 (X, Y)
        -1.0f,  3.0f  // Vertex 3 (X, Y)
    };

    // create vao
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // read shaders from file
    std::string vs = read_file("svo.vert");
    const char* vertexSource = vs.c_str();
    std::string fs = read_file("svo.frag");
    const char* fragmentSource = fs.c_str();

    char buffer[512];
    GLint status;

    // setup vertex shader

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);

    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);

    glGetShaderInfoLog(vertexShader, 512, NULL, buffer);

    std::cout << status << " = status\nlog: " << buffer << "\n";


    // setup fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);

    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);

    glGetShaderInfoLog(fragmentShader, 512, NULL, buffer);

    std::cout << status << " = status\nlog: " << buffer << "\n";

    // create shader program

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    // glBindFragDataLocation ...

    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // specify position vertex attribute
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(posAttrib);

    GLint viewportSize = glGetUniformLocation(shaderProgram, "viewportSize");
    glUniform2f(viewportSize, WIDTH, HEIGHT);

    GLint time = glGetUniformLocation(shaderProgram, "time");

    SDL_Event event;

    bool running = true;
    for (unsigned int tick = 0; running; tick++) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        glDrawArrays(GL_TRIANGLES, 0, 3);
        glUniform1f(time, tick / 60.0f);
        //render_scene(tick);
        SDL_GL_SwapWindow(window);
    }

    // clean up and exit
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
