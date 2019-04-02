all:
	g++ cpu.cpp -lglut -lGLU -lGL -lm
	./a.out
	#nvcc -lglut -lGL -lGLU -o zazen *.cu
	#./zazen

debug:
	g++ cpu.cpp -lglut -lGLU -lGL -lm -O0 -g
	gdb a.out --tui
