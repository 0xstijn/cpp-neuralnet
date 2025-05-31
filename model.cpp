#include "linalg.h"
#include "model.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <vector>

Layer::Layer(int neuron_amount, int prev_neuron_amount, std::string activation_function, bool first) {
    if (first) {
        return;
    }
    if (neuron_amount < 1) {
        throw std::invalid_argument("Layer cannot have less then 1 neuron");
    }
    this->neuron_amount = neuron_amount;

    // More activation functions here :)
    if (activation_function == "relu") {
        this->activation_function = relu;
    }
    else if (activation_function == "softmax") {
        this->activation_function = softmax;
    }
    else {
        throw std::invalid_argument("Invalid activation function in layer");
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
    if ((int) activation_vector.size() != this->neuron_amount) {
        throw std::invalid_argument("Vector doesn't match neuron amount");
    }

    this->activation = this->activation_function(activation_vector);
}


std::vector<double> Layer::relu(std::vector<double> vec) {
    for (int i = 0; i < (int) vec.size(); i++) {
        if (vec[i] < 0) {
            vec[i] = 0;
        }
    }
    return vec;
}


std::vector<double> Layer::softmax(std::vector<double> vec) {
    double vec_sum = 0;
    for (int i = 0; i < (int) vec.size(); i++) {
        vec_sum += vec[i];
    }
    for (int i = 0; i < (int) vec.size(); i++) {
        vec[i] = vec[i] / vec_sum;
    }
    return vec;
}


Model::Model(std::vector<int> dimensions, std::string activation_function, std::string loss_function) {
    if ((int) dimensions.size() < 3) {
        throw std::invalid_argument("Model must have at least 3 layers");
    }

    std::vector<Layer> layers;

    int prev_neuron_amount;
    for (int i = 0; i < (int) dimensions.size(); i++) {
        bool first = i == 0;
        if (first) {
            layers.emplace_back(dimensions[i], 0, "", first);
        }
        else {
            layers.emplace_back(dimensions[i], prev_neuron_amount, activation_function, first);
        }
        prev_neuron_amount = dimensions[i];
    }
    this->layers = layers;
    this->activation_function = activation_function;

    if (loss_function == "categorical_cross_entropy_loss") {
        this->loss_function = categorical_cross_entropy_loss;
    }
    else {
        throw std::invalid_argument("Invalid loss functions for model");
    }
}


double Model::categorical_cross_entropy_loss(std::vector<double> prediction, std::vector<double> target, double epsilon) {
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("Prediction dimensions don't match target dimensions");
    }
    double loss = 0;
    for (int i = 0; i < (int) prediction.size(); i++) {
        loss += -(log(prediction[i] + epsilon) * target[i]);
    }
    return loss;
}