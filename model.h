#ifndef MODEL_H
#define MODEL_H

#include "linalg.h"

class Layer {
public:
    int neuron_amount;
    int prev_neuron_amount;
    bool first;
    Matrix weights;
    std::vector<double> biases;
    std::vector<double> activation;

    Layer(int neuron_amount, int prev_neuron_amount, bool first=false);
};

class Model {
public:
    std::vector<Layer> layers;
    std::string activation_function;
};


#endif