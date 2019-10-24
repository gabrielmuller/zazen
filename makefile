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
	g++ -Wall -Wextra -g -O3 src/construct/main.cpp -o build/construct

prof:
	g++ -g -fprofile-arcs -O3 -Ofast src/render/main.cpp -lSDL2 -o build/render
	g++ -g -fprofile-arcs -O3 src/construct/main.cpp -o build/construct


run_construct:
	cd build; ./construct

run_render:
	cd build; ./render generated.zaz
