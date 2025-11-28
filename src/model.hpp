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
    std::vector<double> (*activation_function)(const std::vector<double>&);
    std::vector<double> biases;
    std::vector<double> activation;
    std::vector<double> pre_activation;

    Layer(int neuron_amount, int prev_neuron_amount, const std::string& activation_function, bool first);
    void activate(const std::vector<double>& activation_vector);
private:
    static std::vector<double> relu(const std::vector<double>& vec);
    static std::vector<double> relu_der(const std::vector<double>& vec);
    static std::vector<double> softmax(const std::vector<double>& vec);
};

class Model {
public:
    std::vector<Layer> layers;
    std::string activation_function;
    std::vector<int> dimensions;

    double (*loss_function)(const std::vector<double>&, const std::vector<double>&, double);

    // Dimensions is signifies how many vectors there are in each layer. Example: {3, 2, 4}
    Model(const std::vector<int>& dimensions, const std::string& activation_function, const std::string& loss_function, const std::vector<Layer>& layers = {});
    std::vector<double> forward(const std::vector<double>& input);

    void save(const std::string& filename);
    static Model init_from_file(const std::string& filename);
private:
    static double categorical_cross_entropy_loss(const std::vector<double>& prediction, const std::vector<double>& target, double epsilon = 0.0000001);
    std::vector<double> categorical_cross_entropy_loss_gradient(const std::vector<double>& prediction, const std::vector<double>& target);
};


#endif
