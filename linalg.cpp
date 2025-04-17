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


class Matrix{
public:
    int rows;
    int columns;

    Matrix(int r, int c, std::vector<std::vector<double>> data) {
        // These checks could be omitted for performance
        if (data.size() != r) {
            throw std::invalid_argument("Dimensions of the data passed in don't match dimensions passed"); 
        } 
        for (std::vector<double> row: data) {
            if (row.size() != r) {
                throw std::invalid_argument("Dimensions of the data passed in don't match dimensions passed"); 
            } 
        }
        rows = r;
        columns = c;
        matrix = data;

        // read
        int operator[](size_t index) const {
            return matrix[index];
        }

        // For writing
        int& operator[](size_t index) {
            return matrix[index];
        }
    }    
    // Returns a transposed copy but doesn't transpose the matrix itself
    Matrix Matrix::transposed() const {
        std::vector<std::vector<double>> transposed_matrix(columns, std::vector<double>(rows));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; i < columns; j++) {
                transposed_matrix[j][i] = data[i][j];
            }
        }
        return transposed_matrix;
    };

    // matrix multiplication
    Matrix operator*(const Matrix m1, const Matrix m2)
    {
        if (m1.rows != m2.columns) {
            std::osstringstream error_message << "cannot do matrix multiplication with matrix of "
                                              << m1.rows << " rows and matrix of " << m2.columns << " columns" << std::endl;
            throw std::invalid_argument(error_message.str());
        }

        Matrix m2_transposed = m2.transposed();
        std::vector<std::vector<double>> matrix_data(m1.rows, std::vector<double>(m1.columns));
        for (int i; i < m1.rows; i++) {
            for (int j; j < m2.colums; j++) {
                matrix_data[i][j] = dot_product(m1[i], m2[j]);
            }
        }
        return Matrix(m1.rows, m2.columns, matrix_data);
    }

private:
    std::vector<std::vector<double>> matrix;
};