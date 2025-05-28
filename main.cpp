#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include "linalg.h"
#include "main.h"
#include "utils.h"




Layer::Layer(int nodes, int prev_nodes) {

    
  int width = nodes;
  int height = prev_nodes;

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
    return output_1;
   

}






