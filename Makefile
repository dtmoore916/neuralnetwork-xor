CC=g++
CFLAGS=-I.
DEPS = simple.h
OBJ = simple.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

simple: $(OBJ)
	gcc -lstdc++ -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f simple.o simple
#rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
