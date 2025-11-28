#include "linalg.h"


class Layer{
    public:
        // Constructor
        Layer(int neuron_amount, int prev_neuron_amount, std::string activation = "reLu");
        // Activation function
        std::string activation;
        Matrix weights;
        std::vector<double> biases;
        //Forward pass
        std::vector<double> forward(std::vector<double> input);

       

};
class Model{
    public:
        // Constructor
        Model(std::vector<int> layer_sizes, std::vector<std::string> activations);
        //Full layer vector
        std::vector<Layer> layers;
        // Activation function
        std::string activation_function;
        int model_size;
        // Forward pass       
        std::vector<double> feed_forward(std::vector<double> input);
        //Loss
        std::vector<double> loss(std::vector<double> output, std::vector<double> target);
        


};
std::vector<double> add_vectors(std::vector<double> v1, std::vector<double> v2);

