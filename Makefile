PROGRAM=xor
CC=g++

CFLAGS=-I.

DEPS := $(wildcard *.h)

OBJ := $(patsubst %.c,%.o,$(wildcard *.c))
OBJ += $(patsubst %.cpp,%.o,$(wildcard *.cpp))

#$(warning $(OBJ))

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(PROGRAM): $(OBJ)
	gcc -lstdc++ -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(PROGRAM) $(OBJ)

