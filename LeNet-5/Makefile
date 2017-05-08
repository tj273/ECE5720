all:p1

p1:main.c lenet.c lenet.h
	gcc -std=c99 -fopenmp -w main.c lenet.c lenet.h -o cnn -lm
	./cnn
clean:
	rm cnn
