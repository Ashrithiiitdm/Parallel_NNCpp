CXX = g++
CXXFLAGS = -std=c++11 -Wall -Werror -pedantic -O3 -fno-inline -fprofile-arcs -ftest-coverage
TARGET = nn

# Match .cpp files recursively in src and helpers directories
SRCS = main.cpp $(wildcard src/*.cpp) $(wildcard helpers/*.cpp)

# Define object files based on the .cpp files
OBJS = $(SRCS:.cpp=.o)

# Default target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule for compiling .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean object files and the target
clean:
	rm -f $(TARGET) $(OBJS)
	rm -f $(wildcard *.gcda) $(wildcard *.gcno)
	rm -f gmon.out
