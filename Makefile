#Compiler
CC=gcc

#Compiler flags
CFLAGS=-pg -fprofile-arcs -ftest-coverage -lm

#Compile and run the program

all:
	$(CC) main.c -o main $(CFLAGS)
	./main

#For flat profile
	gprof main gmon.out > gprof_profile.txt

#For gcov results
	gcov main.c

clean:
	rm -f main gmon.out main.o
	rm -f *.gcda *.gcno