all: build

build: construct render run_construct run_render

run: build run_construct run_render

clean:
	rm -rf build
	mkdir build

render:
	g++ -Wall -Wextra -O3 -Ofast src/render/main.cpp -fopenmp -IC:\mingw64\include -LC:\mingw64\lib -lSDL2 -lglew32 -lopengl32 -o build/render
	cp src/render/shaders/* build/

construct:
	g++ -Wall -Wextra -O3 -Ofast src/construct/main.cpp -o build/construct

debug:
	g++ -Wall -Wextra -g -Og src/construct/main.cpp -o build/construct

prof:
	g++ -g -fprofile-arcs -O3 -Ofast src/render/main.cpp -lSDL2 -o build/render
	g++ -g -fprofile-arcs -O3 src/construct/main.cpp -o build/construct


run_construct:
	cd build; ./construct

run_render:
	cd build; ./render bunny.zaz

bench:
	cd build; ./render generated.zaz 1280 720 1
	cd build; ./render generated.zaz 640 480 1
	cd build; ./render generated.zaz 1280 720 2
	cd build; ./render generated.zaz 640 480 2
	cd build; ./render generated.zaz 1280 720 4
	cd build; ./render generated.zaz 640 480 4
	cd build; ./render generated.zaz 1280 720 8
	cd build; ./render generated.zaz 640 480 8
