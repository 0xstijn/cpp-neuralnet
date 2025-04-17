#include <iostream>
#include <stdexcept>
#include <vector>
#include "linalg.h"

double dot_product(std::vector<double> v1, std::vector<double>) {
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
            return data[index];
        }

        // For writing
        int& operator[](size_t index) {
            return data[index];
        }


    }    
private:
    std::vector<std::vector<double>> matrix;
};