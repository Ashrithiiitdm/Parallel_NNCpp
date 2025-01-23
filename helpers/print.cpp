#include <iostream>
#include <vector>

void print_vector(std::vector<double> v){
    std::cout << "[";
    int n = v.size();

    for (int i = 0; i < n; i++)
    {
        std::cout << v[i];

        if (i != n - 1)
        {
            std::cout << ", ";
        }
    }

    std::cout << "]" << "\n";
}