#define SDL_MAIN_HANDLED
#define GLEW_STATIC

#include <fstream>
#include <ctime>
#include <GL/glew.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "render.cpp"

unsigned char pixels[WIDTH][HEIGHT][4];
const unsigned int UPSCALE = 4;

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

void position_camera(const float time) {
    cam_center.origin = Vector(sin(time)*0.9,
                               cos(time/1.12)*0.9,
                               sin(time/3.21) * 0.9 - 2.1
                               );
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

void GLAPIENTRY glDebugOutput(
        GLenum source,
        GLenum type,
        GLuint id,
        GLenum severity,
        GLsizei length,
        const GLchar* message,
        const void* userParam
        ) {
    //std::cout << message << std::endl;
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
            //SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            1200, 100,
            WIDTH, HEIGHT,
            SDL_WINDOW_OPENGL
    );

    SDL_GLContext context = SDL_GL_CreateContext(window);

    glewExperimental = GL_TRUE;
    glewInit();

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(glDebugOutput, 0);

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

    // Create intermediate frame buffer objects
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);


    // Attach texture to FBO
    GLuint colorTexture;
    glGenTextures(1, &colorTexture);
    glBindTexture(GL_TEXTURE_2D, colorTexture);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        WIDTH  / UPSCALE,
        HEIGHT / UPSCALE,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        0
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        colorTexture,
        0
    );  

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "FBO incomplete!";
        return 1;
    }




    // Allocate shaders and assemble program
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint lightShader = glCreateShader(GL_FRAGMENT_SHADER);

    GLuint shaderProgram = glCreateProgram();
    GLuint lightProgram = glCreateProgram();

    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    glAttachShader(lightProgram, vertexShader);
    glAttachShader(lightProgram, lightShader);

    if (
        compile_shader(vertexShader, "svo.vert") &&
        compile_shader(fragmentShader,  "../src/render/shaders/svo.frag") &&
        compile_shader(lightShader, "../src/render/shaders/light.frag")
    ) {
        glLinkProgram(shaderProgram);
        glLinkProgram(lightProgram);
    }


    // Specify position vertex attribute

    glUseProgram(shaderProgram);
    {
        GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
        glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(posAttrib);
    }

    {
        GLint viewportSize = glGetUniformLocation(shaderProgram, "viewportSize");
        GLint modelSize = glGetUniformLocation(shaderProgram, "modelSize");
        glUniform2ui(viewportSize, WIDTH / UPSCALE, HEIGHT / UPSCALE);
        glUniform1ui(modelSize, block->size());
    }

    GLint time = glGetUniformLocation(shaderProgram, "time");
    GLint camPos = glGetUniformLocation(shaderProgram, "camPos");

    glUseProgram(lightProgram);
    {
        GLint viewportSize = glGetUniformLocation(lightProgram, "viewportSize");
        GLint upscale = glGetUniformLocation(lightProgram, "upscale");
        glUniform2ui(viewportSize, WIDTH, HEIGHT);
        glUniform1ui(upscale, UPSCALE);
    }

    SDL_Event event;

    bool running = true;
    float prev_time = clock() / CLOCKS_PER_SEC;

    for (unsigned int tick = 0; running; tick++) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        // First pass
        glUseProgram(shaderProgram);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        position_camera(tick / 60.0);
        glUniform1f(time, 0);
        glUniform3f(
            camPos,
            cam_center.origin.x, cam_center.origin.y, cam_center.origin.z
        );

        glDrawArrays(GL_TRIANGLES, 0, 3);


        // Second pass
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(lightProgram);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        SDL_GL_SwapWindow(window);

    }

    // Clean up and exit
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
