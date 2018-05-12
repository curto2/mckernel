CPP_FILES := $(wildcard ../../src/*.cpp) RBF_Gaussian_Test.cpp
O_FILES = $(addprefix o/, $(notdir $(CPP_FILES:.cpp=.o)))
CC = g++
FLAGS = -std=c++11 -Wall -O3 -msse2 -mavx -USE_SSE2
LIBS = 
E_NAME = RBF_Gaussian_Test

all: makedirs main

clear:
	rm -r o/

o/%.o : ../../src/%.cpp
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS)

o/%.o : %.cpp
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS)

makedirs:
	mkdir -p o
	
main: $(O_FILES)
	$(CC) $(FLAGS) $^ -o $(E_NAME) $(LIBS)
