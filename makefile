all:
	nvcc -lglut -lGL -lGLU -o zazen *.cu
	./zazen
