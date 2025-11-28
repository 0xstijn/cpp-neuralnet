#ifndef LINALG_H
#define LINALG_H

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>

// Function to calculate dot product of two vectors
double dot_product(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> add_vectors(const std::vector<double>& v1, const std::vector<double>& v2);


// Matrix class
class Matrix {
public:
    int rows;
    int columns;

    // Constructor to initialize the matrix
    Matrix(int r, int c, const std::vector<std::vector<double>>& data);
    Matrix();

    // Transpose method that returns a new transposed matrix
    Matrix transposed() const;

    // Matrix multiplication operator
    Matrix operator*(const Matrix& m2);

    // matrix mult for matrix * vector
    std::vector<double> operator*(const std::vector<double>& vec);



    // Matrix addition
    Matrix operator+(const Matrix& m2);

    // Non-const version (allows modification)
    std::vector<double>& operator[](size_t index);

    // Const version (read-only)
    const std::vector<double>& operator[](size_t index) const;



    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

private:
    std::vector<std::vector<double>> matrix;
};

Matrix hadamard(const Matrix& m1, const Matrix& m2);

#endif // LINALG_H