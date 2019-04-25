all:
	g++ -O3 cpu.cpp -lglut -lGLU -lGL -lm
	./a.out > log
	#nvcc -lglut -lGL -lGLU -o zazen *.cu
	#./zazen

debug:
	g++ -O0 -g cpu.cpp -lglut -lGLU -lGL -lm
	gdb a.out --tui
