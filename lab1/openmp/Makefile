CXX       = g++
CXXFLAGS  = -O3 -march=native -std=c++17 -fopenmp

all: jacobi_openmp

jacobi_openmp: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f jacobi_openmp
