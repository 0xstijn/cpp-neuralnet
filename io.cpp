// This file contains functions for loading and saving matrixes to file
#include "model.h"
#include <iostream>
#include <fstream>

void Model::save(std::string filename) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file for saving model");
    }

    file << "LAYERS\n";
    std::vector<int> dimensions = this->dimensions;
    for (int i = 0; i < (int) dimensions.size(); i++) {
        file << dimensions[i] << "\n";
    }

    for (int i = 0; i < (int) dimensions.size(); i++) {
        file << "L" << i << "\n";
        file << "W\n" << this->layers[i].weights;
        file << "B\n";
        std::vector<double> biases = this->layers[i].biases;
        for (int j = 0; j < (int) biases.size(); j++) {
            file << biases[j] << " ";
        }
        file << "\n";
    }
    file << "END";
}