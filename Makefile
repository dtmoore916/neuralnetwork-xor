PROGRAM=xor
CC=g++

CFLAGS=-I.

DEPS := $(wildcard *.h)

SRC := $(wildcard *.c)
SRC += $(wildcard *.cpp)

ASTYLE_BACKUPS := $(patsubst %.c,%.c.orig,$(wildcard *.c))
ASTYLE_BACKUPS += $(patsubst %.cpp,%.cpp.orig,$(wildcard *.cpp))
ASTYLE_BACKUPS += $(patsubst %.h,%.h.orig,$(wildcard *.h))

OBJ := $(patsubst %.c,%.o,$(wildcard *.c))
OBJ += $(patsubst %.cpp,%.o,$(wildcard *.cpp))

#$(warning $(OBJ))

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(PROGRAM): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: format
format: $(SRC) $(DEPS)
	astyle --indent=tab=4 $^

.PHONY: clean
clean:
	rm -f $(PROGRAM) $(OBJ) $(ASTYLE_BACKUPS)

