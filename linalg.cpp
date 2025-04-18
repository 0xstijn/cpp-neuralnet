#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include "linalg.h"

double dot_product(std::vector<double> v1, std::vector<double> v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Can only calculate dot product of vectors of same size");
    }

    double product = 0;
    for (int i = 0; i < v1.size(); i++) {
        product += v1[i] * v2[i];
    }
    return product;
}


Matrix::Matrix(int r, int c, std::vector<std::vector<double>> data) {
    // These checks could be omitted for performance
    if (data.size() != r) {
        throw std::invalid_argument("Matrix :: Dimensions of the data passed in don't match dimensions passed"); 
    } 
    for (std::vector<double> row: data) {
        if (row.size() != c) {
            throw std::invalid_argument("Matrix :: Dimensions of the data passed in don't match dimensions passed columns"); 
        } 
    }
    rows = r;
    columns = c;
    matrix = data;

}    

// Non-const version
std::vector<double>& Matrix::operator[](size_t index) {
    return matrix[index];  // this->is fine, because matrix is non-const here
}


// Returns a transposed copy but doesn't transpose the matrix itself
Matrix Matrix::transposed() {
    std::vector<std::vector<double>> transposed_matrix(columns, std::vector<double>(rows));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            transposed_matrix[j][i] = this->matrix[i][j];
        }
    }
    Matrix transposed_m(this->columns, this->rows, transposed_matrix);
    return transposed_m;
};

// matrix multiplication
Matrix Matrix::operator*(Matrix m2)
{
    if (this->columns != m2.rows) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }

    Matrix m2_transposed = m2.transposed();
    std::vector<std::vector<double>> matrix_data(this->rows, std::vector<double>(m2_transposed.rows));
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < m2_transposed.rows; j++) {
            matrix_data[i][j] = dot_product((*this)[i], m2_transposed[j]);
        } 
    }

    std::cout << this->columns << m2.rows << std::endl;    
    Matrix output = Matrix(this->rows, m2.columns, matrix_data);
    return output;
}

std::ostream& operator<<(std::ostream& os, Matrix& matrix) {
    for (const auto& row : matrix.matrix) {
        for (double val : row) {
            os << val << " ";
        }
        os << "\n";
    }
    return os;
}