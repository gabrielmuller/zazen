#define SDL_MAIN_HANDLED
#define GLEW_STATIC

#include <fstream>
#include <GL/glew.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "render.cpp"

unsigned char pixels[WIDTH][HEIGHT][4];

std::string read_file(std::string filename) {
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

    return content;
}

void position_camera(unsigned int tick) {
    const float time = tick / 60.0F;
    cam_center.origin = Vector(sin(time)*0.9,
                               sin(time/3.21) * 0.9 + 1.1,
                               cos(time/1.12)*0.9);
}

bool compile_shader(GLuint shader, std::string filename) {
    // Read shaders from file
    std::string source_str = read_file(filename);
    const char* source = source_str.c_str();

    char buffer[512];
    GLint status;

    // Setup vertex shader
    glShaderSource(shader, 1, &source, NULL);

    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    glGetShaderInfoLog(shader, 512, NULL, buffer);

    if (status != 1) {
        std::cout << "Error compiling '" << filename << "':\n" << buffer;
        return false;
    }
    return true;
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

    // Create Shader Storage Buffer Object that will store SVO data
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        block->byte_size(),
        block->data(),
        GL_DYNAMIC_READ
    );
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // Unbind

    // Create vertex array object
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create vertex buffer object
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Allocate shaders and assemble program
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    if (
        compile_shader(vertexShader, "svo.vert") &&
        compile_shader(fragmentShader, "svo.frag")
    ) {
        glLinkProgram(shaderProgram);
    }

    glUseProgram(shaderProgram);

    // Specify position vertex attribute
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(posAttrib);

    GLint viewportSize = glGetUniformLocation(shaderProgram, "viewportSize");
    GLint time = glGetUniformLocation(shaderProgram, "time");

    SDL_Event event;

    bool running = true;
    for (unsigned int tick = 0; running; tick++) {
        // XXX: this is a dev tool. Remove later
        if (!(tick % 30))
        if (
            compile_shader(fragmentShader, "svo.frag")
        ) {
            glLinkProgram(shaderProgram);
            glUniform2ui(viewportSize, WIDTH, HEIGHT);
        }
        // XXX end

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        glUniform1f(time, tick / 60.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        SDL_GL_SwapWindow(window);
    }

    // Clean up and exit
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
