# CS61064 - High Perfomance Parallel Programming
# OpenMP/MPI - Assignment 1
# Parallel Image morphing by affine transformations
# 
# Author: Utkarsh Patel (18EC35034)

# top-level rule to compile the whole program.
all: main

# program is made of several source files.
main: main.o utils.o atx.o
	g++ main.o utils.o atx.o -o main -fopenmp `pkg-config --cflags --libs opencv4`

# rule for file "main.o".
main.o: main.cpp utils.hpp atx.hpp
	g++ -g -Wall -c main.cpp -fopenmp `pkg-config --cflags --libs opencv4`

# rule for file "file1.o".
utils.o: utils.cpp utils.hpp
	g++ -g -Wall -c utils.cpp `pkg-config --cflags --libs opencv4`

# rule for file "file2.o".
atx.o: atx.cpp atx.hpp
	g++ -g -Wall -c atx.cpp `pkg-config --cflags --libs opencv4`

# rule for cleaning files generated during compilations.
clean:
	/bin/rm -f main main.o utils.o atx.o