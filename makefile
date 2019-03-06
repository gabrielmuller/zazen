all:
	nvcc -lglut -lGL -lGLU -o zazen *.cu
	./zazen

debug:
	nvcc -g -G -lglut -lGL -lGLU -o zazen *.cu
	cuda-gdb --tui zazen
