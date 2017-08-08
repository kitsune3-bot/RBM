#pragma once
#include <Eigen/Core>
class BatchNormalization
{
public:
    Eigen::VectorXf average;
    Eigen::VectorXf variance;

public:
    BatchNormalization();
    ~BatchNormalization();
};

