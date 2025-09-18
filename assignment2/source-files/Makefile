CC=gcc
CXX=g++

CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
# Preprocessing flags, not CPP flags
CPPFLAGS=

LIBS=-pthread

# First target is the default target
all: problem1 problem2 problem3

BIN=./bin
$(shell mkdir -p $(BIN))

.DEFAULT_TARGET: all

# $@ is a variable that contains the target name.
problem1: 210874-prob1.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

problem2.1: 210874-prob2_wo_opt.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

problem2.2: 210874-prob2_wo_fs.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

problem2.3: 210874-prob2_wo_ts.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)


problem2: 210874-prob2.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

problem3: 210874-prob3.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(BIN)/$@.out $(LIBS)

# The target is not a filename, always run the rule even if there is a file
.PHONY: clean
clean:
	rm -f -- bin *.out compile_commands.json
