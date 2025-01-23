#pragma once
#include<iostream>

double get_random(){
    return (rand() / (double)RAND_MAX) * 2 - 1;
}