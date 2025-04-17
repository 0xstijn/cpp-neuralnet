#include "linalg.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<vector<double>> m1_data = {
        {1.0, 3.0, 4.0},
        {6.0, 7.0, 3.0}
    }; 
    Matrix m1 = Matrix(2, 3, m1_data);

    Matrix transposed_m1 = m1.transposed();

    vector<vector<double>> m2_data {
        {2},
        {4},
        {1}
    };
    Matrix m2 = Matrix(3, 1, m2_data);
    cout << m1;
    cout << transposed_m1 << endl;
    cout << dot_product({4, 5, 6}, {1, 2, 3}) << endl;

    Matrix m3 = m1 * m2;
    cout << "  " << m3 << "    " << endl;

    cout << m1[1][1];
    return 0;
}