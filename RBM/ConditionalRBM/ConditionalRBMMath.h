#pragma once
#include <cmath>

class ConditionalRBMMath
{
public:
    ConditionalRBMMath();
    ~ConditionalRBMMath();

    static double sigmoid(double x);
};

// シグモイド関数
inline double ConditionalRBMMath::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
