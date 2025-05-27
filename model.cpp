#include "linalg.h"
#include "model.h"
#include "utils.h"
#include <iostream>
#include <vector>

Layer::Layer(int neuron_amount, int prev_neuron_amount, bool first) {
    if (first) {
        return;
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