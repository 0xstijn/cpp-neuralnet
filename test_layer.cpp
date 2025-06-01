#include "linalg.h"
#include <iostream>
#include <vector>
#include "main.h"

using namespace std;

void print_vector(vector<double> vec) {
    cout << "[";
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i];
        if (i < vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}

int main(){
    //Test 1 layer
    Layer layerr = Layer(5, 3);
    vector<double> input = {1.0, 2.0,3};
    vector<double> output = layerr.forward(input);
    cout << "Output of the layer: ";
    print_vector(output);

    //Test a model
    std::vector<int> layers = {3, 5, 2,5,8,9,14,2};
    std::vector<std::string> activations = {"reLu", "reLu", "reLu", "reLu", "reLu", "reLu", "sigmoid"};


    Model model = Model(layers, activations);

    //Example input and outpt
    vector<double> model_input = {1.0, 2.0, 3.0};
    std::vector<double> target = {0.5, 0.5};

    vector<double> model_output = model.feed_forward(model_input);
    vector<double> model_loss = model.loss(model_output, target);
    
    cout << "Output of the model: ";
    print_vector(model_output);
    cout << "Loss of the model: ";
    print_vector(model_loss);


    return 0;
    
};
