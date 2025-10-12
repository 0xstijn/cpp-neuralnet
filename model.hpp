#ifndef __cplusplus
#error "This header must be compiled as C++"
#endif

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

    Layer(int neuron_amount, int prev_neuron_amount, std::string activation_function, bool first);
    void activate(std::vector<double> activation_vector);
private:
    static std::vector<double> relu(std::vector<double> vec);
    static std::vector<double> softmax(std::vector<double> vec);
};

class Model {
public:
    std::vector<Layer> layers;
    std::string activation_function;
    std::vector<int> dimensions;

    double (*loss_function)(std::vector<double>, std::vector<double>, double);

    // Dimensions is signifies how many vectors there are in each layer. Example: {3, 2, 4}
    Model(std::vector<int> dimensions, std::string activation_function, std::string loss_function, std::vector<Layer> layers = {});
    std::vector<double> forward(std::vector<double> input);

    void save(std::string filename);
    static Model init_from_file(std::string filename);
private:
    static double categorical_cross_entropy_loss(std::vector<double> prediction, std::vector<double> target, double epsilon = 0.0000001);
};


#endif
