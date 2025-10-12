#include "model.hpp"
#include <iostream>

int main() {
    std::vector<int> dim= {5, 8, 5};
    std::cout<< "initialising...\n";
    Model model(dim, "relu", "categorical_cross_entropy_loss");
    std::cout<< "initialised!!!\n";

    std::vector<double> result = model.forward({1, 60, 7, 12, 4});
    for (int i = 0; i < (int) result.size(); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    result = model.forward({200, 60, 100, 0, 0});
    for (int i = 0; i < (int) result.size(); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    result = model.forward({0, 10, 22, 2, 4});
    for (int i = 0; i < (int) result.size(); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "SAVING";
    model.save("model.jesperstijn");

    std::cout << "loading from file\n";
    Model model2 = Model::init_from_file("model.jesperstijn");

    return 0;
}