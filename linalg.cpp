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
    for (int i = 0; i < (int) v1.size(); i++) {
        product += v1[i] * v2[i];
    }
    return product;
}

std::vector<double> add_vectors(std::vector<double> v1, std::vector<double> v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Can only calculate addition of vectors of same size");
    }

    std::vector<double> addition(v1.size());

    for (int i = 0; i < (int) v1.size(); i++) {
        addition[i] = v1[i] + v2[i];
    }
    return addition;
}

Matrix::Matrix() {
    return;
}

Matrix::Matrix(int r, int c, std::vector<std::vector<double>> data) {
    // These checks could be omitted for performance
    if ((int) data.size() != r) {
        throw std::invalid_argument("Matrix :: Dimensions of the data passed in don't match dimensions passed"); 
    } 
    for (std::vector<double> row: data) {
        if ((int) row.size() != c) {
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

Matrix Matrix::operator+(Matrix m2){
    if (this->rows != m2.rows || this->columns != m2.columns) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    std::vector<std::vector<double>> matrix_result(this->rows, std::vector<double>(m2.columns));
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < m2.rows; j++) {
            matrix_result[i][j] = this->matrix[i][j] + m2[i][j];
        } 
    }
    Matrix result_matrix_object =  Matrix(this->rows, this->columns, matrix_result);
    return result_matrix_object;

};

Matrix hadamard(Matrix m1, Matrix m2){
    if(m1.columns != m2.columns){
        throw std::invalid_argument("Dimensoins in columns don't match!");
    };

    if(m1.rows != m2.rows){
        throw std::invalid_argument("Dimensoins in rows don't match!");
    };
    std::vector<std::vector<double>> result(m2.rows, std::vector<double>(m2.columns));
    
    for(int i = 0; i < m1.rows; i++){
        for(int j = 0; j < m1.columns; j++){
            //Volgens mij gaat dit mis omdat we * opnieuw gedefinieerd hebben
            result[i][j] = m1[i][j] * m2[i][j];
        }
    }

    Matrix result_matrix = Matrix(m1.rows, m1.columns, result);
    return result_matrix;
}

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

    Matrix output = Matrix(this->rows, m2.columns, matrix_data);
    return output;
}

std::vector<double> Matrix::operator*(std::vector<double> vec) {
    if (this->columns != (int) vec.size()) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    std::vector<double> result(this->rows, 0);

    for (int i = 0; i < this->rows; i++) {
        result[i] = dot_product((*this)[i], vec);
    }
    return result;
}


std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (const auto& row : matrix.matrix) {
        for (double val : row) {
            os << val << " ";
        }
        os << "\n";
    }
    return os;
}