#pragma once
#include <cmath>

class RBMMath
{
public:
    RBMMath();
    ~RBMMath();

    static double sigmoid(double x);
};

// シグモイド関数
inline double RBMMath::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
