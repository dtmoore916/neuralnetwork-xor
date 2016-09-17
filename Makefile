PROGRAM=xor
CC=g++

MKDIR_P = mkdir -p
OBJ_DIR = _objs

CFLAGS=-I.

DEPS := $(wildcard *.h)
SRC += $(wildcard *.cpp)
OBJ += $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(wildcard *.cpp))

#$(warning $(OBJ))

all: create_directories $(PROGRAM)

$(PROGRAM): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

$(OBJ_DIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: create_directories
create_directories: ${OBJ_DIR}
${OBJ_DIR}:
	${MKDIR_P} ${OBJ_DIR}

.PHONY: format
format: $(SRC) $(DEPS)
	astyle -n --indent=tab=4 $^

.PHONY: clean
clean:
	rm -rf $(PROGRAM) $(OBJ) $(OBJ_DIR)

