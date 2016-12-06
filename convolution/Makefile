CFLAGS = -O3 -Wno-deprecated-gpu-targets
CC = nvcc

all: convolution

convolution: timer.o convolution.o
	$(CC) $(CFLAGS) -o convolution timer.o convolution.o

clean:
	rm -f *.o convolution

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ -c $<
