all:
	g++ -O3 -ffast-math main.cpp -fopenmp -lglut -lGL -lm -o zazen
	./zazen > log

debug:
	g++ -O0 -g main.cpp -lglut -lGLU -lGL -lm
	gdb a.out --tui
