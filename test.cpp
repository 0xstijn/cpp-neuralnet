#include "model.h"
#include <iostream>

int main() {
    std::vector<int> dim= {5, 10, 3};
    Model model(dim, "relu", "categorical_cross_entropy_loss");
    std::vector<double> result = model.forward({1, 60, 7, 12, 4});
    for (int i = 0; i < (int) result.size(); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "HAHAHAHAHAH HAHAHAHAHAHAH HAHAHAH";
    return 0;
}