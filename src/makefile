all:
	g++ -O3 -ffast-math main.cpp -fopenmp -lglut -lGLU -lGL -lm -o zazen
	./zazen > log
	#nvcc -lglut -lGL -lGLU -o zazen *.cu
	#./zazen

debug:
	g++ -O0 -g main.cpp -lglut -lGLU -lGL -lm
	gdb a.out --tui
