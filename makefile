all: build

build: construct render run_construct run_render

run: build run_construct run_render

clean:
	rm -rf build
	mkdir build

render:
	g++ -Wall -Wextra -O3 -Ofast src/render/main.cpp -fopenmp -lSDL2 -o build/render

construct:
	g++ -Wall -Wextra -O3 -Ofast -funroll-all-loops src/construct/main.cpp -o build/construct

debug:
	g++ -O0 -Wall -Wextra -g src/construct/main.cpp -o build/construct
	g++ -O0 -Wall -Wextra -g -Ofast -DHEADLESS src/render/main.cpp -o build/render

prof:
	g++ -g -pg -O3 -Ofast src/render/main.cpp -lSDL2 -o build/render


run_construct:
	cd build; ./construct

run_render:
	cd build; ./render generated.zaz
