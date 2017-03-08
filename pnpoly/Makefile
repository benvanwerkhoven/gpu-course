CFLAGS = -O3
CC = nvcc

#insert the ignore for deprecated gpu targets if using CUDA 8.0
CFLAGS += $(shell if [ `nvcc --version | grep 8.0 | wc -l` != 0 ]; then echo -arch=sm_52 ; fi)

all: pnpoly

pnpoly: pnpoly.o
	$(CC) $(CFLAGS) -o pnpoly pnpoly.o

clean:
	rm -f *.o pnpoly

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ -c $<
