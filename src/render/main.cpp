#define SDL_MAIN_HANDLED
#define GLEW_STATIC

#include <fstream>
#include <ctime>
#include <iostream>
#include <cmath>

#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <GL/glew.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "ray.cpp"
#include "block.cpp"

unsigned int width, height, upscale;
glm::vec3 camPosition(0.123, 0.213, -3.193);
glm::mat3 camRotation(1.0);
float crossSection = 1;

const float camSpeed = 0.02;
const float lookSpeed = 0.06;
const float sectionSpeed = 0.002;
float pitch = 0; // -89 to 89
float yaw = 90; //  0  to 360

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

void setup_tex(GLuint& texture, GLuint attachment, GLenum interp, GLenum format, GLenum internal) {
    glGenTextures(1, &texture);
    
    glActiveTexture(GL_TEXTURE0 + attachment);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        internal,
        width  / upscale,
        height / upscale,
        0,
        GL_RGBA,
        format,
        0
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, interp);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, interp);

    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0 + attachment,
        GL_TEXTURE_2D,
        texture,
        0
    );  
}

int main(int argc, char **argv) {
    unsigned int arg = 1;
    const std::string filename = arg >= argc ? "sponge.zaz" : argv[arg++];

    Block* block = from_file(filename);

    width = arg >= argc ? 1280 :  atoi(argv[arg++]);
    height = arg >= argc ? 720 :  atoi(argv[arg++]);
    upscale = arg >= argc ? 1 :  atoi(argv[arg++]);

    SDL_Init(SDL_INIT_VIDEO);
    atexit(SDL_Quit);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);


    SDL_Window* window = SDL_CreateWindow(
            "zazen",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            width, height,
            SDL_WINDOW_OPENGL
    );

    SDL_GLContext context = SDL_GL_CreateContext(window);
    SDL_SetRelativeMouseMode(SDL_TRUE);

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

    // Create intermediate frame buffer object
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    GLenum drawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_NONE, GL_NONE, GL_COLOR_ATTACHMENT3};
    glDrawBuffers(4, drawBuffers);

    // Attach textures to FBO
    GLuint colorTexture;
    GLuint positionTexture;

    setup_tex(colorTexture, 0, GL_NEAREST, GL_UNSIGNED_BYTE, GL_RGBA);
    setup_tex(positionTexture, 3, GL_NEAREST, GL_FLOAT, GL_RGBA32F);


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
        compile_shader(fragmentShader,  "svo.frag") &&
        compile_shader(lightShader, "light.frag")
    ) {
        /*
        glBindFragDataLocation(shaderProgram, 0, "outColor");
        glBindFragDataLocation(shaderProgram, 1, "outPosition");
        */
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
        glUniform2ui(viewportSize, width / upscale, height / upscale);
        glUniform1ui(modelSize, block->size());

    }

    GLint time = glGetUniformLocation(shaderProgram, "time");
    GLint xSection = glGetUniformLocation(shaderProgram, "xSection");
    GLint camPos = glGetUniformLocation(shaderProgram, "camPos");
    GLint camRot = glGetUniformLocation(shaderProgram, "camRot");

    glUseProgram(lightProgram);
    {
        GLint viewportSize = glGetUniformLocation(lightProgram, "viewportSize");
        GLint upscaleLoc = glGetUniformLocation(lightProgram, "upscale");
        GLint colorLoc = glGetUniformLocation(lightProgram, "colorTexture");
        GLint positionLoc = glGetUniformLocation(lightProgram, "positionTexture");

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glUniform1i(colorLoc, 0);

        glActiveTexture(GL_TEXTURE0 + 3);
        glBindTexture(GL_TEXTURE_2D, positionTexture);
        glUniform1i(positionLoc, 3);

        glUniform2ui(viewportSize, width, height);
        glUniform1ui(upscaleLoc, upscale);

    }

    glDisable(GL_DEPTH_TEST);

    SDL_Event event;

    bool running = true;
    unsigned long long timer = clock();
    glm::vec3 deltaCamPos(0);
    float deltaSection = 0;

    for (unsigned int tick = 0; running; tick++) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_UP:
                    case SDLK_w:
                        deltaCamPos.z = camSpeed;
                        break;
                    case SDLK_DOWN:
                    case SDLK_s:
                        deltaCamPos.z = -camSpeed;
                        break;
                    case SDLK_LEFT:
                    case SDLK_a:
                        deltaCamPos.x = -camSpeed;
                        break;
                    case SDLK_RIGHT:
                    case SDLK_d:
                        deltaCamPos.x = camSpeed;
                        break;
                    case SDLK_LSHIFT:
                        deltaCamPos.y = -camSpeed;
                        break;
                    case SDLK_SPACE:
                        deltaCamPos.y = camSpeed;
                        break;
                    case SDLK_q:
                        deltaSection = -sectionSpeed;
                        break;
                    case SDLK_e:
                        deltaSection = sectionSpeed;
                        break;
                }
            } else if (event.type == SDL_KEYUP) {
                switch (event.key.keysym.sym) {
                    case SDLK_UP:
                    case SDLK_DOWN:
                    case SDLK_w:
                    case SDLK_s:
                        deltaCamPos.z = 0;
                        break;
                    case SDLK_LEFT:
                    case SDLK_RIGHT:
                    case SDLK_a:
                    case SDLK_d:
                        deltaCamPos.x = 0;
                        break;
                    case SDLK_LSHIFT:
                    case SDLK_SPACE:
                        deltaCamPos.y = 0;
                        break;
                    case SDLK_q:
                    case SDLK_e:
                        deltaSection = 0;
                        break;
                }
            } else if (event.type == SDL_MOUSEMOTION) {
                yaw += event.motion.xrel * lookSpeed;
                pitch += event.motion.yrel * lookSpeed;
                yaw = std::fmod(yaw, 360.0);
                if (pitch < -89.0) pitch = -89.0;
                if (pitch >  89.0) pitch =  89.0;
                glm::vec3 pitchDir(
                    0,
                    sin(glm::radians(pitch)),
                    cos(glm::radians(pitch))
                );
                pitchDir = glm::normalize(pitchDir);
                glm::vec3 yawDir(
                    cos(glm::radians(yaw)),
                    0,
                    sin(glm::radians(yaw))
                );
                yawDir = glm::normalize(yawDir);

                camRotation = glm::lookAt(
                    glm::vec3(0),
                    yawDir,
                    glm::vec3(0,1,0)
                ) * glm::lookAt(
                    glm::vec3(0),
                    pitchDir,
                    glm::vec3(0,1,0)
                );
            }
        }

        camPosition += camRotation * deltaCamPos;
        crossSection += deltaSection;
        if (crossSection < 0) crossSection = 0;
        if (crossSection > 1) crossSection = 1;

        // First pass
        glUseProgram(shaderProgram);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glUniform1f(time, 0);
        glUniform1f(xSection, crossSection);
        glUniform3fv(camPos, 1, &camPosition[0]);
        glUniformMatrix3fv(camRot, 1, GL_FALSE, &camRotation[0][0]);

        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Second pass
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glUseProgram(lightProgram);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        SDL_GL_SwapWindow(window);

        if (tick == 500 && false) {
            timer = clock() - timer;
            std::cout << timer << "\n";
            std::cout << width << "X" << height << " " << upscale << "\n"
            << "Average FPS of " << 500.0 / (float(timer) / CLOCKS_PER_SEC) << "\n\n";
            break;
        }

    }

    // Clean up and exit
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
