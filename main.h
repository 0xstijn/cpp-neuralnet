#include "linalg.h"


class Layer{
    public:
        // Constructor
        Layer(int neuron_amount, int prev_neuron_amount);

        Matrix weights;
        std::vector<double> biases;
        //Forward pass
        std::vector<double> forward(std::vector<double> input);
       

};


class Model{
    public:

        Model(int* array, int size);

};