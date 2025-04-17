#ifndef LINALG_H
#define LINALG_H

#include <vector>

double dot_product(const std::vector<double>& v1, const std::vector<double>& v2);

class Matrix {
public:
    int rows;
    int columns;

    Matrix(int r, int c, const std::vector<std::vector<double>>& data);

    std::vector<double>& operator[](size_t index);
    const std::vector<double>& operator[](size_t index) const;

private:
    std::vector<std::vector<double>> matrix;
};

#endif // LINALG_H#ifndef LINALG