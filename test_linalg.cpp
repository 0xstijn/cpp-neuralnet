#include "linalg.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<vector<double>> m1_data = {
        {2, 5},
        {4, 3}
    };
    Matrix m1(2, 2, m1_data);

    vector<vector<double>> m2_data = {
        {5, 5},
        {8, 1}
    };
    Matrix m2(2, 2, m2_data);

    cout << hadamard(m1, m2) << endl;
    cout << m1[0][0] << m1[0][1] << endl;
}