// This file contains functions for loading and saving matrixes to file
#include "model.hpp"
#include <iostream>
#include <fstream>

std::vector<double> parse_line(const std::string& line);

void Model::save(const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file for saving model");
    }

    file << "LAYERS\n";
    const std::vector<int>& dimensions = this->dimensions;
    for (size_t i = 0; i < dimensions.size(); i++) {
        file << dimensions[i] << "\n";
    }
    file << "ENDLAYERS\n";

    for (size_t i = 1; i < dimensions.size(); i++) {
        file << "LAYER\n";
        file << "W\n" << this->layers[i].weights;
        file << "B\n";
        const std::vector<double>& biases = this->layers[i].biases;
        for (size_t j = 0; j < biases.size(); j++) {
            file << biases[j] << " ";
        }
        file << "\n";
    }
    file << "END";
}

Model Model::init_from_file(const std::string& filename) {
    std::ifstream infile(filename);

    if (!infile) {
        std::cerr << "can't open file " << filename << std::endl;
    }

    std::vector<int> dimensions = {};

    std::string line;
    std::getline(infile, line);
    while (line != "ENDLAYERS") {
        if (line != "LAYERS") {
            dimensions.push_back(std::stoi(line));
        }
        std::getline(infile, line);  // verplaatst naar einde van lus
    }

    std::vector<double> bias;
    std::vector<Layer> layers = {};
    Layer layer = Layer(dimensions[0], 0, "relu", true);
    layers.push_back(layer);

    std::getline(infile, line);

    int i = 0;
    do {
        i += 1;
        std::getline(infile, line);
        if (line == "END") {
            break;
        }
        std::vector<std::vector<double>> matrix_values = {};
        do {
            std::getline(infile, line);
            if (line == "W") {
                continue;
            }
            std::vector<double> row = parse_line(line);
            if (row.size() != 0) {
                matrix_values.push_back(row);
            }

        } while (line != "B");

        Matrix weights(dimensions[i], dimensions[i - 1], matrix_values);

        std::getline(infile, line);
        bias = parse_line(line);
        layer = Layer(dimensions[i], dimensions[i-1], "relu", false);
        layer.weights = weights;
        layer.biases = bias;

        layers.push_back(layer);
    } while (line != "END");

    Model model(dimensions, "relu", "categorical_cross_entropy_loss", layers);
    return model;

}

std::vector<double> parse_line(const std::string& line) {
    std::vector<double> parsed = {};
    std::string number_string = "";
    double number;
    for (size_t i = 0; i < line.size(); i++) {
	    if (line[i] == ' ') {
            number = std::stod(number_string);
            parsed.push_back(number);
            number_string = "";
        }
        else {
            number_string += line[i];
        }
    }
    return parsed;
}
