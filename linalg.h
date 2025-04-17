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
    Matrix transposed() const;

    // Matrix multiplication operator
    Matrix operator*(const Matrix m1, const Matrix m2);

    // Access operator for reading (const)
    const std::vector<double>& operator[](size_t index) const;

    // Access operator for writing (non-const)
    std::vector<double>& operator[](size_t index);

private:
    std::vector<std::vector<double>> matrix;
};

#endif // LINALG_H