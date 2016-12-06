CFLAGS = -O3 --Wno-deprecated-gpu-targets
CC = nvcc

all: vector_add

vector_add: timer.o vector_add.o
	$(CC) $(CFLAGS) -o vector_add timer.o vector_add.o

clean:
	rm -f *.o vector_add

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ -c $<
