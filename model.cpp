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

    this->activation = this->activation_function(activation_vector);
}


std::vector<double> Layer::relu(std::vector<double> vec) {
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] < 0) {
            vec[i] = 0;
        }
    }
    return vec;
}


std::vector<double> Layer::softmax(std::vector<double> vec) {
    double vec_sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        vec_sum += vec[i];
    }
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = vec[i] / vec_sum;
    }
    return vec;
}