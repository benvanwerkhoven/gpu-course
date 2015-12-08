CFLAGS = -O3
CC = g++

all: loop_order

loop_order: timer.o loop_order.o
	$(CC) $(CFLAGS) -o loop_order timer.o loop_order.o

clean:
	rm -f *.o loop_order

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ -c $<
