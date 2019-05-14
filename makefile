all: build

build: clean construct render run_construct

run: build run_construct run_render

clean:
	rm -rf build
	mkdir build

render:
	g++ -O3 -ffast-math src/render/main.cpp -fopenmp -lglut -lGL -lm -o build/render

construct:
	g++ -O3 src/construct/main.cpp -fopenmp -o build/construct

debug:
	g++ -O0 -g src/render/main.cpp -lglut -lGLU -lGL -lm -o build/debug
	g++ -O0 -g src/construct/main.cpp

run_construct:
	cd build; ./construct

run_render:
	cd build; ./render brain.zaz
