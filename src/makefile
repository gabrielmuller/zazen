all: clean render.bin

clean:
	rm *.bin
render.bin:
	g++ -O3 -ffast-math render/main.cpp -fopenmp -lglut -lGL -lm -o render.bin
	./render.bin > log

construct.bin:
	g++ -O3 construct/main.cpp -fopenmp -o construct.bin
	./construct.bin

debug:
	g++ -O0 -g main.cpp -lglut -lGLU -lGL -lm
	gdb a.out --tui
