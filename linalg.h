#ifndef LINALG_H
#define LINALG_H

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>

// Function to calculate dot product of two vectors
double dot_product(std::vector<double> v1, std::vector<double> v2);

// Matrix class
class Matrix {
public:
    int rows;
    int columns;

    // Constructor to initialize the matrix
    Matrix(int r, int c, std::vector<std::vector<double>> data);

    // Transpose method that returns a new transposed matrix
    Matrix transposed();

    // Matrix multiplication operator
    Matrix operator*(Matrix m2);

    // Non-const version (allows modification)
    std::vector<double>& operator[](size_t index);


    friend std::ostream& operator<<(std::ostream& os, Matrix& matrix);

private:
    std::vector<std::vector<double>> matrix;
};

#endif // LINALG_H