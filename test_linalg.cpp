#include "linalg.h"
#include <iostream>
#include <vector>

int main() {
    std::vector<std::vector<double>> m1_data = {
        {1.0, 3.0, 4.0},
        {6.0, 7.0, 3.0}
    }; 
    Matrix m1 = Matrix(2, 3, m1_data);
    std::cout << m1;
    return 0;
}