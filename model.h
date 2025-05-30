#ifndef MODEL_H
#define MODEL_H

#include "linalg.h"

class Layer {
public:
    int neuron_amount;
    int prev_neuron_amount;
    bool first;
    Matrix weights;
    std::vector<double> (*activation_function)(std::vector<double>);
    std::vector<double> biases;
    std::vector<double> activation;

    Layer(int neuron_amount, int prev_neuron_amount, bool first=false, std::string activation_function);
    void activate(std::vector<double> activation_vector);
private:
    static std::vector<double> relu(std::vector<double> vec);
    static std::vector<double> softmax(std::vector<double> vec);
};

class Model {
public:
    std::vector<Layer> layers;
    std::string activation_function;
};


#endif