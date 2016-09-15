CC=g++
CFLAGS=-I.
DEPS = simple.h network.h
OBJ = simple.o network.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

simple: $(OBJ)
	gcc -lstdc++ -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f simple.o network.o simple
#rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
