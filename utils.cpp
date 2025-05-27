#include "utils.h"
#include <iostream>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());

double random_double(double minimum, double maximum) {
    std::uniform_real_distribution<> dist(minimum, maximum);
    double random_value = dist(gen);
    return random_value;
}
