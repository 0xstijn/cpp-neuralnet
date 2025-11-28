#include "linalg.h"
#include "model.hpp"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <vector>

Layer::Layer(int neuron_amount, int prev_neuron_amount, const std::string& activation_function, bool first) {
    if (neuron_amount < 1) {
        throw std::invalid_argument("Layer cannot have less then 1 neuron");
    }
    this->neuron_amount = neuron_amount;

    if (first) {
        return;
    }

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

void Layer::activate(const std::vector<double>& activation_vector) {
    if ((int) activation_vector.size() != this->neuron_amount) {
        throw std::invalid_argument("Vector doesn't match neuron amount");
    }

    this->pre_activation = activation_vector;
    this->activation = this->activation_function(activation_vector);
}


std::vector<double> Layer::relu(const std::vector<double>& vec) {
    std::vector<double> result = vec;
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] < 0) {
            result[i] = 0;
        }
    }
    return result;
}
    
static std::vector<double> relu_der(const std::vector<double>& vec) {
    std::vector<double> result = vec;
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] <= 0) {
            result[i] = 0;
        }
        else {
            result[i] = 1;
        }
    }
    return result;
}


std::vector<double> Layer::softmax(const std::vector<double>& vec) {
    std::vector<double> result = vec;
    double max = result[0];
    for (size_t i = 1; i < result.size(); i++) {
        if (result[i] > max) {
            max = result[i];
        }
    }
    double vec_sum = 0;
    for (size_t i = 0; i < result.size(); i++) {
        result[i] -= max;
        vec_sum += std::exp(result[i]);
    }
    for (size_t i = 0; i < result.size(); i++) {
        result[i] = std::exp(result[i]) / vec_sum;
    }
    return result;
}


Model::Model(const std::vector<int>& dimensions, const std::string& activation_function, const std::string& loss_function, const std::vector<Layer>& layers) {
    if (dimensions.size() < 3) {
        throw std::invalid_argument("Model must have at least 3 layers");
    }
    this->dimensions = dimensions;


    int prev_neuron_amount = 0;
    std::vector<Layer> model_layers = layers;
    if (model_layers.size() == 0) {
        for (size_t i = 0; i < dimensions.size(); i++) {
            bool first = i == 0;
            bool last = i == dimensions.size() - 1;
            if (first) {
                model_layers.emplace_back(dimensions[i], 0, "", first);
            }
            else if (last) {
                model_layers.emplace_back(dimensions[i], prev_neuron_amount, "softmax", first);
            }
            else {
                model_layers.emplace_back(dimensions[i], prev_neuron_amount, activation_function, first);
            }
            prev_neuron_amount = dimensions[i];
        }
    }
    else if (model_layers.size() != dimensions.size()) {
        throw std::invalid_argument("Layers and dimensions must match");
    }
    this->layers = model_layers;
    this->activation_function = activation_function;

    if (loss_function == "categorical_cross_entropy_loss") {
        this->loss_function = categorical_cross_entropy_loss;
    }
    else {
        throw std::invalid_argument("Invalid loss functions for model");
    }
}


std::vector<double> Model::forward(const std::vector<double>& input) {
    const Layer& first_layer = this->layers[0];
    if ((int) input.size() != first_layer.neuron_amount) {
        throw std::invalid_argument("input doesnt match first layer dimension");
    }

    size_t layer_amount = this->layers.size();
    std::vector<double> previous_activation = input;

    for (size_t i = 1; i < layer_amount; i++) {
        Layer& current_layer = this->layers[i];
        std::vector<double> next = add_vectors(current_layer.weights * previous_activation, current_layer.biases);
        current_layer.activate(next);
        previous_activation = current_layer.activation;
    }
    return previous_activation;
}


double Model::categorical_cross_entropy_loss(const std::vector<double>& prediction, const std::vector<double>& target, double epsilon) {
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("Prediction dimensions don't match target dimensions");
    }
    double loss = 0;
    for (size_t i = 0; i < prediction.size(); i++) {
        loss += -(log(prediction[i] + epsilon) * target[i]);
    }
    return loss;
}

std::vector<double> Model::categorical_cross_entropy_loss_gradient(const std::vector<double>& prediction, const std::vector<double>& target) {
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("Prediction dimensions don't match target dimensions");
    }
    std::vector<double> gradient(prediction.size());
    for (size_t i = 0; i < prediction.size(); i++) {
        gradient[i] = prediction[i] - target[i];
    }
    return gradient;
}