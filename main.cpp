#include <iostream>
using namespace std;


//Represents 1 layer of the model
class Layer {
  public:

    int nodes; //Amount of nodes of the layer
    string activation; //Which activation function to use

    //Paramters need for forward pass and backprop
    float z = 0; //Output vector before activation
    float a = 0; //Output vector after activation

    //New layer needs amount of nodes and activation function
    Layer(int layer_nodes, string activation_function = "ReLu") {

      nodes = layer_nodes;
      activation = activation_function; 
    }

    void init_matrices(int next_layer_nodes){
      float weight[nodes][next_layer_nodes]; //Init weight matrix of shape(nodes, next_layer_nodes)
      float bias[1][next_layer_nodes]; //Init bias vector

      //These are both zero matrix and zero vectors for now. We need to give each index a random
      //value between e.g. -1 and 1 or make them gaussian distributed (doesnt really matter in practice)
      //Thought about doing it with a double for loop but may not be most efficient
    }

    void forward(int input_vector){
      //input_vector is the resulting output_vector from previous layer
      z = weight * input_vector + bias; //First term is matrix multiplication
      a = activation(z);
      return a;
    }
};

class Model {
  public:

    int layer_count = 1;
    Layer total_layer[1] = {Layer(0, "ReLu")};
    Model(int input_size, string activation = "ReLu"){

      //Initializes the input vector
      Layer layer = Layer(input_size, activation);
      total_layer[0] = layer;
    };

    void add(int layer_nodes, string activation = "ReLu"){
      //Add layer to the model
      layer_count += 1;
      Layer new_layer = Layer(layer_nodes, activation);
      Layer new_total_layer[layer_count] = {total_layer, new_layer};


    };




    

};


int main() {
  cout << "Hello World!";
  return 0;
};



