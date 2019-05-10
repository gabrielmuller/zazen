all: clean construct.bin render.bin

clean:
	rm -f *.bin

render.bin:
	g++ -O3 -ffast-math render/main.cpp -fopenmp -lglut -lGL -lm -o render.bin

construct.bin:
	g++ -O3 construct/main.cpp -fopenmp -o construct.bin

debug:
	g++ -O0 -g construct/main.cpp -lglut -lGLU -lGL -lm
	gdb a.out --tui
