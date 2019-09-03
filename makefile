all: build

build: construct render run_construct

run: build run_construct run_render

clean:
	rm -rf build
	mkdir build

render:
	g++ -O3 -Ofast src/render/main.cpp -fopenmp -lSDL2 -o build/render

construct:
	g++ -O3 src/construct/main.cpp -o build/construct

debug:
	g++ -O0 -g src/render/main.cpp -lglut -lGLU -lGL -lm -o build/render
	g++ -O0 -g src/construct/main.cpp -o build/construct

prof:
	g++ -g -pg -O3 -Ofast src/render/main.cpp -fopenmp -lSDL2 -o build/render


run_construct:
	cd build; ./construct

run_render:
	cd build; ./render generated.zaz
