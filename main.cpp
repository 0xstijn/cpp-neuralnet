#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cmath>
#include "linalg.h"
#include "main.h"
#include "utils.h"



double reLu(double input) {
    return input > 0 ? input : 0;
}

double sigmoid(double input) {
    return 1 / (1 + exp(-input));
}

double activation_functions(double input, std::string type) {
    if (type == "reLu") {
        return reLu(input);
    } else if (type == "sigmoid") {
        return sigmoid(input);
    } else {
        throw std::invalid_argument("Unknown activation function type: " + type);
    }
    
}

std::vector<double> activate(std::vector<double> activation_vector, std::string activation) {

    std::vector<double> activated_vector = std::vector<double>(activation_vector.size());
    for(int i = 0; i < activation_vector.size(); i++) {
        activated_vector[i] = activation_functions(activation_vector[i], activation);
    }
    return activated_vector;
}

std::vector<double> add_vector(std::vector<double> v1, std::vector<double> v2){
    if (v1.size() != v2.size()){
        throw std::invalid_argument("Vectors must have same size!");
    
    }
    std::vector<double> result = std::vector<double>(v1.size());
    for (int i = 0; i < v1.size(); i++){
        result[i] = v1[i] + v2[i];
    }
    return result;
}


//Begin of layer clase
Layer::Layer(int nodes, int prev_nodes, std::string activation) {

    
  int width = nodes;
  int height = prev_nodes;
  this->activation = activation;

  std::vector<std::vector<double>> weights_data(width, std::vector<double>(height, 0));
  for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
          weights_data[i][j] = random_double();
      }
  }

  Matrix weights(width, height, weights_data);
  this->weights = weights;
  std::vector<double> biases(nodes, 0);
  this->biases = biases;

}

std::vector<double> Layer::forward(std::vector<double> input) {
    if (input.size() != this->weights.columns) {
        throw std::invalid_argument("Input size does not match layer weights dimensions");
    }
    // Perform matrix multiplication and add biases
    std::vector<double> output_1 = this->weights * input;
    std::vector<double> output_2 = add_vector(output_1, this->biases);
    std::vector<double> output_3 = activate(output_2, this->activation);
    return output_3;
}


//Begin of Model class
Model::Model(std::vector<int> layer_sizes, std::vector<std::string> activations) {

    std::vector<Layer> layers;
    int model_size;
    if(layer_sizes.size() != activations.size() + 1) {
        throw std::invalid_argument("Number of activations must be one less than number of layers");
    }
    if(layer_sizes.size() < 2) {
        throw std::invalid_argument("Model must have at least two layers");
    }

    for(int i = 1; i< layer_sizes.size(); i++) {
        //Create the full model of layers
        layers.push_back(Layer(layer_sizes[i], layer_sizes[i-1], activations[i-1]));
    }
    
    this->layers = layers;
    this->model_size = layers.size();

}

std::vector<double> Model::feed_forward(std::vector<double> input) {
    std::vector<double> output = input;
    for(int i = 0; i < this->model_size; i++) {
        output = this->layers[i].forward(output);
    }
    return output;
}

std::vector<double> Model::loss(std::vector<double> output, std::vector<double> target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output and target vectors must have the same size");
    }
    std::vector<double> loss_vector(output.size());
    for (int i = 0; i < output.size(); i++) {
        //Using mean squared error for the moment gonna add modularity later
        loss_vector[i] = 0.5 * std::pow(output[i] - target[i], 2);
    }
    return loss_vector;
}



