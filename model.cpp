#include "linalg.h"
#include "model.h"
#include "utils.h"
#include <iostream>
#include <vector>

Layer::Layer(int neuron_amount, int prev_neuron_amount, bool first, std::string activation_function) {
    if (first) {
        return;
    }

    // More activation functions here :)
    if (activation_function == "relu") {
        this->activation_function = relu;
    }
    
    int weight_rows = neuron_amount;
    int weight_columns = prev_neuron_amount;
    // Initialise the correct dimensions
    std::vector<std::vector<double>> weights_data(weight_rows, std::vector<double>(weight_columns, 0));

    for (int i = 0; i < weight_rows; i++) {
        for (int j = 0; j < weight_columns; j++) {
            weights_data[i][j] = random_double();
        }
    }
    
    Matrix weights(weight_rows, weight_columns, weights_data);
    this->weights = weights;

    std::vector<double> biases(neuron_amount, 0);
    this->biases = biases;
}

void Layer::activate(std::vector<double> activation_vector) {
    if (activation_vector.size() != this->neuron_amount) {
        throw std::invalid_argument("Vector doesn't match neuron amount");
    }

    for (int i = 0; i < activation_vector.size(); i++) {
        activation_vector[i] = this->activation_function(activation_vector[i]);
    }
    this->activation = activation_vector;
}


double Layer::relu(double x) {
    if (x < 0) {
        return 0;
    }
    else {
        return x;
    }
}