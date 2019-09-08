#define SDL_MAIN_HANDLED

#ifndef HEADLESS

#include <SDL2/SDL.h>
#include <SDL2/SDL_render.h>
SDL_Texture* texture;
SDL_Renderer* renderer;

#endif

#include "render.cpp"


unsigned char pixels[WIDTH][HEIGHT][4];

void render_scene(unsigned int tick) {
#ifndef HEADLESS
    SDL_RenderClear(renderer);
#endif

    const float time = tick / 60.0F;
    cam_center.origin = Vector(sin(time)*0.9,
                               sin(time/3.21) * 0.9 + 1.1,
                               cos(time/1.12)*0.9);

    #pragma omp parallel for schedule(dynamic)
    for (unsigned int i = 0; i < WIDTH; i++) {
        for (unsigned int j = 0; j < HEIGHT; j++) {
            render(pixels[j][i], i, j);
        }
    }

#ifndef HEADLESS
    SDL_UpdateTexture(
        texture,
        nullptr,
        pixels,
        WIDTH * 4
    );

    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
#endif
}


int main(int argc, char **argv) {
    const int arg = 1;

    if (arg >= argc) {
        std::cout << "Please specify an input file.\n";
        return 1;
    }

    block = from_file(argv[arg]);

#ifdef HEADLESS
    for (unsigned int tick = 0; ; tick++) render_scene(tick);
#else

    SDL_Init(SDL_INIT_EVERYTHING);
    atexit(SDL_Quit);

    SDL_Window* window = SDL_CreateWindow(
            "zazen",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            WIDTH, HEIGHT,
            SDL_WINDOW_SHOWN
    );

    renderer = SDL_CreateRenderer(
        window, 
        -1,
        SDL_RENDERER_ACCELERATED
    );

    texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_BGR888,
        SDL_TEXTUREACCESS_STREAMING,
        WIDTH, HEIGHT
    );

    SDL_Event event;


    bool running = true;
    for (unsigned int tick = 0; running; tick++) {
        while (SDL_PollEvent(&event)) if (event.type == SDL_QUIT) running = false;
        render_scene(tick);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
#endif
}
